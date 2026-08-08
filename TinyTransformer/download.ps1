# 下载 GPT-2 模型文件到 ./weights 目录
# 用法:
#   pwsh -File download.ps1                 # 默认 gpt2 small
#   pwsh -File download.ps1 -Variant gpt2-xl
#   pwsh -File download.ps1 -OutDir D:\models\gpt2
# 若执行策略拦截, 用:  pwsh -ExecutionPolicy Bypass -File download.ps1

[CmdletBinding()]
param(
    [ValidateSet('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')]
    [string]$Variant = 'gpt2',

    [string]$OutDir,

    [switch]$UseMirror
)

$ErrorActionPreference = 'Stop'

# 切到脚本所在目录
Set-Location -Path $PSScriptRoot

# 1) 无需额外依赖（download.py 用标准库 urllib 直接 HTTP 下载，绕过 Xet）

# 2) 国内镜像（用 -UseMirror 开关或已设置 $env:HF_ENDPOINT 时生效）
if ($UseMirror -and -not $env:HF_ENDPOINT) {
    $env:HF_ENDPOINT = 'https://hf-mirror.com'
}

# 3) 构造参数并下载
$pyArgs = @('download.py', '--variant', $Variant)
if ($OutDir) { $pyArgs += @('--out', $OutDir) }

Write-Host "Running: py $($pyArgs -join ' ')" -ForegroundColor Cyan
py @pyArgs
exit $LASTEXITCODE
