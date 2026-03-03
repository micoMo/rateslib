# Git Fork Workflow

## Remotes
- `origin`   → your fork: https://github.com/micoMo/rateslib
- `upstream` → original:  https://github.com/attack68/rateslib

## Push your own changes
```bash
git add .
git commit -m "your message"
git push origin main
```

## Sync when attack68 releases a new version
```bash
git fetch upstream
git merge upstream/main
# fix any conflicts if needed
git push origin main
```

## Check if upstream has new commits
```bash
git fetch upstream
git log main..upstream/main --oneline
```
