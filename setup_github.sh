#!/bin/bash
# Setup script to connect your local repository to GitHub

echo "=============================================="
echo "GitHub Repository Setup Script"
echo "=============================================="
echo ""

# Get user input
read -p "Enter your GitHub username: " GITHUB_USER
read -p "Enter your repository name (e.g., btc-ml-trading-bot): " REPO_NAME

# Confirm
echo ""
echo "Your GitHub repository URL will be:"
echo "https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
read -p "Is this correct? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo "Setup cancelled."
    exit 1
fi

echo ""
echo "Setting up git remote..."

# Remove old remote if exists
git remote remove origin 2>/dev/null

# Add new remote
git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo "✓ Remote added successfully!"
echo ""

# Show remotes
echo "Current remotes:"
git remote -v
echo ""

# Ask about branch
echo "You have commits on branch: claude/ml-trading-bot-backtest-011CV5wsvBg87iMRx52YQ4Mm"
echo ""
echo "Choose an option:"
echo "1) Push to 'main' branch (recommended for GitHub)"
echo "2) Push to current branch name"
read -p "Enter choice (1 or 2): " BRANCH_CHOICE

if [ "$BRANCH_CHOICE" = "1" ]; then
    echo ""
    echo "Creating and pushing to 'main' branch..."
    git checkout -b main 2>/dev/null || git checkout main
    git merge claude/ml-trading-bot-backtest-011CV5wsvBg87iMRx52YQ4Mm --no-edit

    echo ""
    echo "Pushing to GitHub..."
    git push -u origin main

    if [ $? -eq 0 ]; then
        echo ""
        echo "=============================================="
        echo "✓ SUCCESS! Code pushed to GitHub!"
        echo "=============================================="
        echo ""
        echo "Visit your repository at:"
        echo "https://github.com/$GITHUB_USER/$REPO_NAME"
        echo ""
    else
        echo ""
        echo "⚠ Push failed. You may need to authenticate."
        echo "See GITHUB_SETUP.md for authentication help."
    fi
else
    echo ""
    echo "Pushing to current branch..."
    git push -u origin claude/ml-trading-bot-backtest-011CV5wsvBg87iMRx52YQ4Mm

    if [ $? -eq 0 ]; then
        echo ""
        echo "=============================================="
        echo "✓ SUCCESS! Code pushed to GitHub!"
        echo "=============================================="
        echo ""
        echo "Visit your repository at:"
        echo "https://github.com/$GITHUB_USER/$REPO_NAME"
        echo ""
        echo "Note: You may want to set this as the default branch on GitHub."
    else
        echo ""
        echo "⚠ Push failed. You may need to authenticate."
        echo "See GITHUB_SETUP.md for authentication help."
    fi
fi
