# 设置 Dataset 路径
$datasetPath = ".\Dataset"

# 遍历所有 .git 目录
Get-ChildItem -Path $datasetPath -Recurse -Directory -Force | Where-Object { $_.Name -eq ".git" } | ForEach-Object {

  $repo = $_.Parent.FullName
  Write-Host ("🔍 检查仓库: " + $repo)

  # 删除锁文件
  $locks = Get-ChildItem -Path $_.FullName -Recurse -Filter "*.lock" -ErrorAction SilentlyContinue
  foreach ($lock in $locks) {
    Remove-Item -Force -ErrorAction SilentlyContinue $lock.FullName
  }
  Write-Host ("🗝 删除锁文件: " + $locks.Count)

  # 修复权限
  $userString = $env:USERNAME + ":(OI)(CI)F"
  icacls $_.FullName /grant $userString /T | Out-Null

  # 修复 HEAD 文件
  $headPath = Join-Path $_.FullName "HEAD"
  if (Test-Path $headPath) {
    $head = Get-Content $headPath -ErrorAction SilentlyContinue
    if ($head -match '^[0-9a-f]{40}$') {
      Write-Host "⚠️ 修复 HEAD (从提交哈希改为分支指针)"
      'ref: refs/heads/master' | Set-Content $headPath
    }
  }

  # 设置 core.logAllRefUpdates=false
  git -C $repo config --local core.logAllRefUpdates false
  $val = git -C $repo config --local core.logAllRefUpdates
  Write-Host ("✅ core.logAllRefUpdates 当前值: " + $val)

  # 打印 Git 状态
  try {
    git -C $repo status | Out-Host
    Write-Host ("🎯 仓库正常: " + $repo)
  }
  catch {
    Write-Host ("❌ 仓库异常: " + $repo)
  }

  Write-Host ("--------------------------------------`n")
}
