param(
    [string]$HipOgaRoot = $env:ONNXRUNTIME_GENAI_HOME,
    [string]$BinaryDir,
    [string]$ExpectedSha256File,
    [version]$MinimumOrtVersion = [version]"1.27.0"
)

$ErrorActionPreference = "Stop"

function Resolve-HipOgaBin {
    param([string]$Root)

    if ([string]::IsNullOrWhiteSpace($Root)) {
        throw "Set -HipOgaRoot or ONNXRUNTIME_GENAI_HOME to the OGA/HIP EP SDK root."
    }

    $resolved = Resolve-Path -LiteralPath $Root
    $bin = Join-Path $resolved "bin"
    if (Test-Path -LiteralPath $bin -PathType Container) {
        return (Resolve-Path -LiteralPath $bin).Path
    }

    return $resolved.Path
}

function Read-ExpectedHashes {
    param([string]$Path)

    $hashes = @{}
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $hashes
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
            $name = Split-Path -Leaf ($Matches[2] -replace '/', '\')
            $hashes[$name.ToLowerInvariant()] = $Matches[1].ToLowerInvariant()
        }
    }

    return $hashes
}

$candidateBin = if ($BinaryDir) {
    (Resolve-Path -LiteralPath $BinaryDir).Path
} else {
    Resolve-HipOgaBin -Root $HipOgaRoot
}

$requiredDlls = @(
    "onnxruntime-genai.dll",
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "amdgpu-ep.dll",
    "hip-backend.dll"
)
$expectedHashes = Read-ExpectedHashes -Path $ExpectedSha256File

foreach ($dll in $requiredDlls) {
    $path = Join-Path $candidateBin $dll
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required HIP OGA runtime DLL: $path"
    }

    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($expectedHashes.ContainsKey($dll.ToLowerInvariant()) -and
        $hash -ne $expectedHashes[$dll.ToLowerInvariant()]) {
        throw "Unexpected SHA256 for $dll`: expected $($expectedHashes[$dll.ToLowerInvariant()]), got $hash"
    }

    Write-Output "file=$path|sha256=$hash"
}

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class NativeLibraryLoader {
    public const uint LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100;
    public const uint LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000;

    [DllImport("kernel32", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hFile, uint dwFlags);

    [DllImport("kernel32", SetLastError=true, CharSet=CharSet.Ansi)]
    public static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);
}

public delegate IntPtr OrtGetApiBaseDelegate();
public delegate IntPtr OrtGetVersionStringDelegate();
"@

$loadFlags = [NativeLibraryLoader]::LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR -bor
    [NativeLibraryLoader]::LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
$loadedHandles = @()
foreach ($dll in $requiredDlls) {
    $path = Join-Path $candidateBin $dll
    $handle = [NativeLibraryLoader]::LoadLibraryEx($path, [IntPtr]::Zero, $loadFlags)
    if ($handle -eq [IntPtr]::Zero) {
        $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw "LoadLibraryEx failed for $path with Win32 error $errorCode"
    }
    $loadedHandles += $handle
}

$loaded = Get-Process -Id $PID | Select-Object -ExpandProperty Modules |
    Where-Object { $requiredDlls -contains $_.ModuleName } |
    Sort-Object ModuleName

foreach ($module in $loaded) {
    $moduleHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $module.FileName).Hash.ToLowerInvariant()
    Write-Output "loaded=$($module.ModuleName)|path=$($module.FileName)|sha256=$moduleHash"
}

$ortModule = $loaded | Where-Object { $_.ModuleName -ieq "onnxruntime.dll" } | Select-Object -First 1
if (-not $ortModule) {
    throw "onnxruntime.dll was not loaded while loading onnxruntime-genai.dll"
}
if ((Resolve-Path -LiteralPath (Split-Path -Parent $ortModule.FileName)).Path -ne $candidateBin) {
    throw "onnxruntime.dll loaded from $($ortModule.FileName), expected $candidateBin"
}

$ortHandle = [NativeLibraryLoader]::LoadLibraryEx($ortModule.FileName, [IntPtr]::Zero, $loadFlags)
$apiBaseSymbol = [NativeLibraryLoader]::GetProcAddress($ortHandle, "OrtGetApiBase")
if ($apiBaseSymbol -eq [IntPtr]::Zero) {
    throw "onnxruntime.dll does not export OrtGetApiBase: $($ortModule.FileName)"
}

$getApiBase = [Runtime.InteropServices.Marshal]::GetDelegateForFunctionPointer(
    $apiBaseSymbol,
    [type][OrtGetApiBaseDelegate])
$apiBase = $getApiBase.Invoke()
$getVersionStringSymbol = [Runtime.InteropServices.Marshal]::ReadIntPtr($apiBase, [IntPtr]::Size)
$getVersionString = [Runtime.InteropServices.Marshal]::GetDelegateForFunctionPointer(
    $getVersionStringSymbol,
    [type][OrtGetVersionStringDelegate])
$ortVersionString = [Runtime.InteropServices.Marshal]::PtrToStringAnsi($getVersionString.Invoke())

Write-Output "loaded_ort_version=$ortVersionString"
if ([version]$ortVersionString -lt $MinimumOrtVersion) {
    throw "Loaded ORT $ortVersionString is older than required $MinimumOrtVersion for OGA/API-v27"
}

Write-Output "PASS: HIP OGA runtime DLL provenance is package-local and ORT >= $MinimumOrtVersion"
