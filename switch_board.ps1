param(
    [string]$BoardType = ""
)

$firmware_dir = "sis_fall_firmware_inference"
$sdkconfig = "$firmware_dir\sdkconfig"
$main_c = "$firmware_dir\main\sis_fall_firmware_inference.c"

if (-not (Test-Path $sdkconfig)) {
    Write-Host "Không tìm thấy file sdkconfig. Vui lòng chạy idf.py build một lần để tạo."
    exit
}

if ($BoardType -eq "ch343") {
    Write-Host "=> Đang cấu hình cho ESP32-S3 THƯỜNG (CH343/CP2102)..."
    
    # Đổi sdkconfig về UART0
    $c = Get-Content $sdkconfig
    $c = $c -replace '^CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y$', '# CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG is not set'
    $c = $c -replace '^# CONFIG_ESP_CONSOLE_UART_DEFAULT is not set$', 'CONFIG_ESP_CONSOLE_UART_DEFAULT=y'
    $c | Set-Content $sdkconfig

    # Đổi cờ C code về 0
    $c_code = Get-Content $main_c
    $c_code = $c_code -replace '#define USE_NATIVE_USB 1', '#define USE_NATIVE_USB 0'
    $c_code | Set-Content $main_c

    Write-Host "[Xong] Mạch thường đã sẵn sàng. Gõ lệnh sau để nạp:"
    Write-Host "       cd sis_fall_firmware_inference ; idf.py build flash"

} elseif ($BoardType -eq "seeed") {
    Write-Host "=> Đang cấu hình cho ESP32-S3 SEEED STUDIO (Native USB)..."
    
    # Đổi sdkconfig về Native USB
    $c = Get-Content $sdkconfig
    $c = $c -replace '^# CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG is not set$', 'CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y'
    $c = $c -replace '^CONFIG_ESP_CONSOLE_UART_DEFAULT=y$', '# CONFIG_ESP_CONSOLE_UART_DEFAULT is not set'
    $c | Set-Content $sdkconfig

    # Đổi cờ C code về 1
    $c_code = Get-Content $main_c
    $c_code = $c_code -replace '#define USE_NATIVE_USB 0', '#define USE_NATIVE_USB 1'
    $c_code | Set-Content $main_c

    Write-Host "[Xong] Mạch Seeed đã sẵn sàng. Gõ lệnh sau để nạp:"
    Write-Host "       cd sis_fall_firmware_inference ; idf.py build flash"

} else {
    Write-Host "============================================="
    Write-Host "CÔNG CỤ ĐỔI NHANH CẤU HÌNH CHO 2 LOẠI MẠCH"
    Write-Host "============================================="
    Write-Host "Cách dùng:"
    Write-Host "1. Nếu cắm mạch thường (CH343): .\switch_board.ps1 ch343"
    Write-Host "2. Nếu cắm mạch Seeed (USB):    .\switch_board.ps1 seeed"
}
