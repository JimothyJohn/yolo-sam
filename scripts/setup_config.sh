#!/bin/bash
# Description:
# Writes the environment variables to AWS SSM Parameter Store.

set -e

# Default values
ENV="prod"
DOMAIN_NAME=""
HOSTED_ZONE_ID=""
STACK_NAME=""
APP_NAME="ophanim"

# Usage function
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -d, --domain <domain>        Domain name"
    echo "  -z, --hosted-zone <id>       Hosted Zone ID"
    echo "  -s, --stack-name <name>      Stack Name (Required)"
    echo "  -a, --app-name <name>        App Name (default: ophanim)"
    echo "  -e, --env <env>              Environment (default: prod)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --domain ophanim.advin.io --hosted-zone Z0123456789ABCDEF --stack-name ophanim-prod --app-name ophanim --env prod"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--domain) DOMAIN_NAME="$2"; shift ;;
        -z|--hosted-zone) HOSTED_ZONE_ID="$2"; shift ;;
        -s|--stack-name) STACK_NAME="$2"; shift ;;
        -a|--app-name) APP_NAME="$2"; shift ;;
        -e|--env) ENV="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Validation
if [ -z "$STACK_NAME" ]; then
    echo "Error: Stack Name is required."
    usage
fi

echo "Putting parameters into SSM for Stack: $STACK_NAME ($ENV)..."

# Construct paths using Stack Name
DOMAIN_PARAM="/$APP_NAME/$STACK_NAME/domain_name"
ZONE_PARAM="/$APP_NAME/$STACK_NAME/hosted_zone_id"

if [ -n "$DOMAIN_NAME" ]; then
    echo "  $DOMAIN_PARAM -> $DOMAIN_NAME"
    # Use Standard parameter type for publicly visible values like domain name
    aws ssm put-parameter --name "$DOMAIN_PARAM" --value "$DOMAIN_NAME" --type String --overwrite
else
    echo "  Skipping Domain Name update (not provided)"
fi

if [ -n "$HOSTED_ZONE_ID" ]; then
    echo "  $ZONE_PARAM -> $HOSTED_ZONE_ID"
    # Hosted Zone ID is technically public if you know the name servers, but safe to treat as sensitive if desired.
    aws ssm put-parameter --name "$ZONE_PARAM" --value "$HOSTED_ZONE_ID" --type String --overwrite
else
    echo "  Skipping Hosted Zone ID update (not provided)"
fi

echo "✅ Configuration stored in SSM!"
