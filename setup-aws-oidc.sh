#!/bin/bash

# Setup AWS OIDC for GitHub Actions
# This script creates the OIDC identity provider and IAM role for GitHub Actions

set -e

# Configuration
GITHUB_REPO="${GITHUB_REPOSITORY:-cagataycali/devduck}"
AWS_REGION="${AWS_REGION:-us-west-2}"
ROLE_NAME="GitHubDevDuckActionsRole" # Replace for each repository
POLICY_NAME="GitHubDevDuckActionsPolicy"

echo "🚀 Setting up AWS OIDC for GitHub Actions..."
echo "Repository: $GITHUB_REPO"
echo "Region: $AWS_REGION"

# Step 1: Create OIDC Identity Provider
echo "📝 Step 1: Creating OIDC Identity Provider..."

# Check if OIDC provider already exists
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):oidc-provider/token.actions.githubusercontent.com" &>/dev/null; then
    echo "✅ OIDC Identity Provider already exists"
else
    echo "Creating OIDC Identity Provider..."
    aws iam create-open-id-connect-provider \
        --url https://token.actions.githubusercontent.com \
        --client-id-list sts.amazonaws.com \
        --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
        --thumbprint-list 1c58a3a8518e8759bf075b76b750d4f2df264fcd

    echo "✅ OIDC Identity Provider created successfully"
fi

# Step 2: Create Trust Policy for the Role
echo "📝 Step 2: Creating trust policy..."

cat > trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):oidc-provider/token.actions.githubusercontent.com"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "token.actions.githubusercontent.com:sub": "repo:${GITHUB_REPO}:*"
                },
                "StringEquals": {
                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                }
            }
        }
    ]
}
EOF

# Step 3: Create IAM Role
echo "📝 Step 3: Creating IAM Role..."

if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    echo "⚠️  Role $ROLE_NAME already exists, updating trust policy..."
    aws iam update-assume-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-document file://trust-policy.json
else
    echo "Creating IAM Role..."
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file://trust-policy.json \
        --description "Role for GitHub Actions OIDC authentication"
fi

# Step 4: Create and attach permissions policy
echo "📝 Step 4: Creating permissions policy..."

cat > permissions-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:*",
                "bedrock-runtime:*",
                "bedrock-agent:*",
                "bedrock-agent-runtime:*",
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "sts:GetCallerIdentity"
            ],
            "Resource": "*"
        }
    ]
}
EOF

# Create or update the policy
if aws iam get-policy --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$POLICY_NAME" &>/dev/null; then
    echo "⚠️  Policy $POLICY_NAME already exists, updating..."
    aws iam create-policy-version \
        --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$POLICY_NAME" \
        --policy-document file://permissions-policy.json \
        --set-as-default
else
    echo "Creating permissions policy..."
    aws iam create-policy \
        --policy-name "$POLICY_NAME" \
        --policy-document file://permissions-policy.json \
        --description "Permissions for GitHub Actions to access AWS services"
fi

# Attach policy to role
echo "📝 Step 5: Attaching policy to role..."
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$POLICY_NAME"

# Cleanup temporary files
rm -f trust-policy.json permissions-policy.json

# Get the role ARN
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add this role ARN to your GitHub repository variables:"
echo "   Variable name: AWS_ROLE_ARN"
echo "   Variable value: $ROLE_ARN"
echo ""
echo "2. Remove the following secrets from your GitHub repository:"
echo "   - AWS_ACCESS_KEY_ID"
echo "   - AWS_SECRET_ACCESS_KEY"
echo "   - AWS_SESSION_TOKEN"
echo ""
echo "3. The GitHub Actions workflow will be updated automatically."
echo ""
echo "Role ARN: $ROLE_ARN"