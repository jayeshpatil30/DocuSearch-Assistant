"""Deploy (or update) the Document Search index-jobs stack. Idempotent: re-run to update.

    uv run python deploy/index-jobs/deploy.py --image specsafe-index-jobs:events-20261117 \
        --specsafe-principal arn:aws:iam::<acct>:user/<service-account> [--with-e2e] \
        --subnet-ids subnet-00000000 --vpc-id vpc-00000000

``--specsafe-principal``: the IAM user or role SpecSafe signs its calls to the sync API
with (the feature's service account). Only it is granted invoke on that function; left
out on an update, the deployed principal is kept. Without ``--image`` the image already
deployed is kept (e.g. to change only the principal).

``--subnet-ids``/``--vpc-id``: the functions that talk to Atlas run in that private
subnet, so they reach Atlas from the NAT gateway's fixed IP (example-vpc:
43.205.74.169), which must be on the Atlas IP Access List.

Run from the repo root after building the image (see Dockerfile). Steps:

1. Push the local image to ECR (``specsafe-index-jobs``) and pin it by digest (skipped
   without ``--image``: the deployed image is kept).
2. Create or update the CloudFormation stack (``template.yaml``) and wait for it. The
   Atlas connection string is ``SEARCH_MONGO_URI`` from ``.env``, passed as the NoEcho
   ``MongoUri`` parameter and set on the functions as their ``SEARCH_MONGO_URI``.
3. Turn on "Send notifications to Amazon EventBridge" on the source bucket, keeping
   every notification already configured there.

Region: the SOURCE BUCKET's (ap-south-1 for demo-docsearch-bucket) -- S3 delivers its events
only to EventBridge in its own region. Embeddings still use Titan in ``--bedrock-region``
(us-east-1), the model the collections were built with.

``--with-e2e`` adds an isolated E2E target (own collection, own catalogue collection,
own S3 prefix) used by ``scripts/e2e_index_events.py``; it never touches TENANT_A or TENANT_B data.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, WaiterError
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# The E2E target: isolated from TENANT_A/TENANT_B. Kept in step with scripts/e2e_index_events.py.
E2E_TARGET = {
    "collection": "E2E_DocumentSearch_DEV",
    "bucket": "demo-docsearch-bucket",
    "prefix": "docsearch-e2e/",
    "tenant_codes": ["E2E"],
    "metadata_database": "DOCSEARCH_DEMO",
    "metadata_collection": "AttachmentMetaData_E2E",
}


def step(msg: str) -> None:
    print(f"\n== {msg}", flush=True)


def push_image(session: boto3.Session, local: str, repo: str) -> str:
    ecr = session.client("ecr")
    try:
        uri = ecr.describe_repositories(repositoryNames=[repo])["repositories"][0]["repositoryUri"]
    except ecr.exceptions.RepositoryNotFoundException:
        uri = ecr.create_repository(
            repositoryName=repo, imageScanningConfiguration={"scanOnPush": True}
        )["repository"]["repositoryUri"]
    token = ecr.get_authorization_token()["authorizationData"][0]
    user, password = base64.b64decode(token["authorizationToken"]).decode().split(":", 1)
    registry = token["proxyEndpoint"]
    subprocess.run(
        ["docker", "login", "--username", user, "--password-stdin", registry],
        input=password.encode(), check=True, capture_output=True,
    )
    tag = local.split(":", 1)[1] if ":" in local else "latest"
    remote = f"{uri}:{tag}"
    subprocess.run(["docker", "tag", local, remote], check=True)
    subprocess.run(["docker", "push", remote], check=True)
    digest = ecr.describe_images(repositoryName=repo, imageIds=[{"imageTag": tag}])[
        "imageDetails"
    ][0]["imageDigest"]
    pinned = f"{uri}@{digest}"
    print("image:", pinned)
    return pinned


def mongo_uri() -> str:
    uri = dotenv_values(ROOT / ".env").get("SEARCH_MONGO_URI") or ""
    if not uri.startswith("mongodb"):
        sys.exit("SEARCH_MONGO_URI is missing from .env")
    return uri


def stack_exists(session: boto3.Session, stack: str) -> bool:
    try:
        session.client("cloudformation").describe_stacks(StackName=stack)
        return True
    except ClientError:
        return False


def deploy_stack(session: boto3.Session, stack: str, params: dict[str, str | None]) -> dict[str, str]:
    """Create or update the stack; a parameter whose value is None keeps its deployed value."""
    cf = session.client("cloudformation")
    body = (HERE / "template.yaml").read_text(encoding="utf-8")
    parameters = [
        {"ParameterKey": k, "UsePreviousValue": True} if v is None
        else {"ParameterKey": k, "ParameterValue": v}
        for k, v in params.items()
    ]
    exists = stack_exists(session, stack)
    try:
        if exists:
            cf.update_stack(StackName=stack, TemplateBody=body, Parameters=parameters,
                            Capabilities=["CAPABILITY_IAM"])
            waiter = cf.get_waiter("stack_update_complete")
        else:
            cf.create_stack(StackName=stack, TemplateBody=body, Parameters=parameters,
                            Capabilities=["CAPABILITY_IAM"], OnFailure="ROLLBACK")
            waiter = cf.get_waiter("stack_create_complete")
        print("waiting for", stack, "...", flush=True)
        waiter.wait(StackName=stack, WaiterConfig={"Delay": 15, "MaxAttempts": 120})
    except ClientError as exc:
        if "No updates are to be performed" not in str(exc):
            raise
        print("stack already up to date")
    except WaiterError:
        events = cf.describe_stack_events(StackName=stack)["StackEvents"][:25]
        for e in events:
            if "FAILED" in e["ResourceStatus"]:
                print(f"  {e['LogicalResourceId']}: {e.get('ResourceStatusReason')}")
        raise
    outputs = cf.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in outputs}


def enable_eventbridge(session: boto3.Session, bucket: str) -> None:
    s3 = session.client("s3")
    config = s3.get_bucket_notification_configuration(Bucket=bucket)
    config.pop("ResponseMetadata", None)
    if "EventBridgeConfiguration" in config:
        print(f"{bucket}: EventBridge notifications already on")
        return
    config["EventBridgeConfiguration"] = {}
    s3.put_bucket_notification_configuration(Bucket=bucket, NotificationConfiguration=config)
    print(f"{bucket}: EventBridge notifications turned on (other notifications kept)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default="",
                   help="local image, e.g. specsafe-index-jobs:events-20261117 (default: keep the deployed one)")
    p.add_argument("--stack", default="docsearch-dev")
    p.add_argument("--region", default="ap-south-1", help="the source bucket's region")
    p.add_argument("--bucket", default="demo-docsearch-bucket")
    p.add_argument("--specsafe-principal", default="",
                   help="the IAM user/role ARN SpecSafe signs with -- the service account "
                        "(default: keep the deployed one; on create, this AWS account)")
    p.add_argument("--bedrock-region", default="us-east-1")
    p.add_argument("--architecture", default="arm64", choices=["arm64", "x86_64"])
    p.add_argument("--with-e2e", action="store_true", help="add the isolated E2E test target")
    p.add_argument("--subnet-ids", default="",
                   help="private subnet(s), comma-separated, behind a NAT whose IP Atlas allows")
    p.add_argument("--vpc-id", default="", help="the VPC of --subnet-ids")
    args = p.parse_args()

    session = boto3.Session(region_name=args.region)
    exists = stack_exists(session, args.stack)
    # Without --specsafe-principal an update keeps the granted caller; never widen it silently.
    principal: str | None = args.specsafe_principal or None
    if principal is None and not exists:
        account = session.client("sts").get_caller_identity()["Account"]
        principal = f"arn:aws:iam::{account}:root"

    uri = mongo_uri()
    step("1/3 image -> ECR")
    if args.image:
        image_uri: str | None = push_image(session, args.image, "specsafe-index-jobs")
    elif exists:
        image_uri = None
        print("no --image: keeping the deployed image")
    else:
        sys.exit("--image is required to create the stack")

    targets = ""
    if args.with_e2e:
        targets = json.dumps({
            "TENANT_A": {"collection": "TENANT_A_DocumentSearch_DEV", "metadata_database": "TENANT_A_DEV"},
            "TENANT_B": {"collection": "TENANT_B_DocumentSearch_DEV", "metadata_database": "TENANT_B_DEV"},
            "E2E": E2E_TARGET,
        })
    step("2/3 CloudFormation")
    outputs = deploy_stack(session, args.stack, {
        "ImageUri": image_uri,
        "MongoUri": uri,
        "SourceBucketArns": f"arn:aws:s3:::{args.bucket}",
        "SourceBucketNames": args.bucket,
        "SpecSafePrincipalArn": principal,
        "BedrockRegion": args.bedrock_region,
        "Architecture": args.architecture,
        "IndexTargetsJson": targets,
        "SubnetIds": args.subnet_ids,
        "VpcId": args.vpc_id,
    })
    step("3/3 S3 -> EventBridge")
    enable_eventbridge(session, args.bucket)

    step("done")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
    print(f"  logs: /aws/lambda/{args.stack}-index-events, -sync-api, -index-jobs ({args.region})")


if __name__ == "__main__":
    main()
