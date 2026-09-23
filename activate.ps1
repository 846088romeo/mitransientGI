# Script de activación rápida para Mitransient
# Uso: .\activate.ps1

Write-Host "Activando entorno de Python..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

$llvmPath = "C:\Program Files\LLVM\bin\LLVM-C.dll"
if (-not (Test-Path $llvmPath)) {
	Write-Error "No se encontró LLVM-C.dll en: $llvmPath"
	exit 1
}

$env:DRJIT_LIBLLVM_PATH = $llvmPath

if (-not (Test-Path $env:DRJIT_LIBLLVM_PATH)) {
	Write-Error "La ruta configurada en DRJIT_LIBLLVM_PATH no es válida: $env:DRJIT_LIBLLVM_PATH"
	exit 1
}

Write-Host "Entorno completamente activado!" -ForegroundColor Green
