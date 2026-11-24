# Draws the three icons the input method shows on the taskbar (cpp\tip\*.ico), in the style of
# Windows' own input methods so they sit well next to them:
#   icon_profile.ico  the input method's own: a framed white tile with U+5100 (yi)
#   icon.ico          the mode button in mixed mode: U+5169 (liang), white with a thin dark outline
#   icon_en.ico       the mode button in English mode: U+82F1 (ying), the same way
# The drawing is in tools\tray_icons.cs. Needs the Microsoft JhengHei font (part of Windows).
#   powershell -ExecutionPolicy Bypass -File tools\make_tray_icons.ps1
# (installer\app.ico and settings\ui\logo.svg still come from tools\make_icons.py.)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$tip = Join-Path $root "cpp\tip"
Add-Type -AssemblyName System.Drawing
Add-Type -Path (Join-Path $PSScriptRoot "tray_icons.cs") -ReferencedAssemblies System.Drawing

$sizes = 16, 20, 24, 32, 40, 48, 64, 256
# (code points, not the characters: Windows PowerShell reads a script without a BOM as ANSI)
$yi = [string][char]0x5100
$liang = [string][char]0x5169
$ying = [string][char]0x82F1
$icons = [ordered]@{
    "icon_profile.ico" = { param($size) [IconDraw]::Tile($yi, $size) }
    "icon.ico"         = { param($size) [IconDraw]::Mode($liang, $size, $false) }
    "icon_en.ico"      = { param($size) [IconDraw]::Mode($ying, $size, $false) }
}
foreach ($name in $icons.Keys) {
    $frames = New-Object 'System.Collections.Generic.List[System.Drawing.Bitmap]'
    foreach ($size in $sizes) { $frames.Add((& $icons[$name] $size)) }
    [IconDraw]::SaveIco((Join-Path $tip $name), $frames)
    Write-Host "wrote cpp\tip\$name"
}
