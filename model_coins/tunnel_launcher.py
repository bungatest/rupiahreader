# -*- coding: utf-8 -*-
"""
Tunnel Launcher otomatis untuk Rupiah Reader.
Menyediakan koneksi HTTPS gratis (Cloudflare Quick Tunnel) agar kamera HP dapat diakses
tanpa perlu install konfigurasi rumit atau mendaftar akun.
"""

import os
import sys
import time
import socket
import re
import shutil
import subprocess
import urllib.request
import urllib.error

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
LOCAL_SERVER_URL = "http://127.0.0.1:8000"

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def check_server_running(url=LOCAL_SERVER_URL):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def download_with_progress(url, dest_path):
    print(f"[INFO] Mengunduh 'cloudflared.exe' dari GitHub resmi...")
    print(f"       Sumber: {url}")
    print(f"       Tujuan: {dest_path}")
    print("[INFO] Ukuran file ~30 MB. Mohon tunggu beberapa saat...\n")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 64 * 1024  # 64 KB
            downloaded = 0
            start_time = time.time()

            with open(dest_path + ".tmp", "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)

                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total_size)
                        bar = "#" * filled + "-" * (bar_len - filled)
                        elapsed = max(0.1, time.time() - start_time)
                        speed_mb = (downloaded / (1024 * 1024)) / elapsed
                        sys.stdout.write(f"\r[{bar}] {pct:5.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB) {speed_mb:.1f} MB/s")
                        sys.stdout.flush()

            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(dest_path + ".tmp", dest_path)
            print("\n\n[SUKSES] cloudflared.exe berhasil diunduh dan siap digunakan!")
            return True
    except Exception as e:
        print(f"\n\n[GAGAL] Unduhan otomatis terhambat: {e}")
        if os.path.exists(dest_path + ".tmp"):
            try:
                os.remove(dest_path + ".tmp")
            except Exception:
                pass
        return False

def copy_to_clipboard(text):
    try:
        subprocess.run(["clip"], input=text.encode("utf-8"), check=True, timeout=2)
        return True
    except Exception:
        return False

def find_cloudflared_binary(script_dir):
    # 1. Cek di PATH
    in_path = shutil.which("cloudflared")
    if in_path:
        return in_path

    # 2. Cek di direktori script / model_coins
    local_bin = os.path.join(script_dir, "cloudflared.exe")
    if os.path.isfile(local_bin) and os.path.getsize(local_bin) > 1000000:
        return local_bin

    # 3. Cek di root frontend
    root_bin = os.path.join(os.path.dirname(script_dir), "cloudflared.exe")
    if os.path.isfile(root_bin) and os.path.getsize(root_bin) > 1000000:
        return root_bin

    return None

def main():
    print("=" * 76)
    print("    RUPIAH READER - HTTPS TUNNEL UNTUK PENGUJIAN KAMERA DI HP")
    print("=" * 76)
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_ip = get_local_ip()

    # Periksa status server port 8000
    if not check_server_running():
        print(">> [PERHATIAN] Server lokal (http://127.0.0.1:8000) belum terdeteksi!")
        print("   Pastikan skrip '1_JALANKAN_SERVER.bat' sudah dibuka di jendela CMD lain.")
        print("   (Tunnel tetap akan dibuka, namun HP baru bisa memuat halaman setelah server menyala)")
        print()

    cf_bin = find_cloudflared_binary(script_dir)

    if not cf_bin:
        print("[INFO] 'cloudflared.exe' belum tersedia di komputer ini.")
        print("       Aplikasi akan mengunduhnya secara otomatis dari Cloudflare (gratis, tanpa akun).")
        print()
        target_path = os.path.join(script_dir, "cloudflared.exe")
        success = download_with_progress(CLOUDFLARED_URL, target_path)

        if success:
            cf_bin = target_path
        else:
            print("\n" + "=" * 76)
            print("PANDUAN ALTERNATIF KONEKSI HP:")
            print("=" * 76)
            print(f"Opsi A: Unduh 'cloudflared-windows-amd64.exe' manual dari browser:")
            print(f"        {CLOUDFLARED_URL}")
            print(f"        Ubah namanya jadi 'cloudflared.exe' dan letakkan di: {script_dir}")
            print()
            print(f"Opsi B: Hubungkan HP ke Wi-Fi yang sama dengan Laptop:")
            print(f"        1. Buka Chrome di HP ke: http://{local_ip}:8000")
            print(f"        2. Jika kamera tidak aktif karena HTTP biasa, buka di Chrome HP:")
            print(f"           chrome://flags/#unsafely-treat-insecure-origin-as-secure")
            print(f"        3. Masukkan 'http://{local_ip}:8000', pilih 'Enabled', lalu Relaunch.")
            print("=" * 76)
            input("\nTekan Enter untuk keluar...")
            return

    print(f"\n[INFO] Menggunakan: {cf_bin}")
    print("[INFO] Membuka secure tunnel ke http://127.0.0.1:8000 ...")
    print("[INFO] Sedang meminta public HTTPS URL dari Cloudflare Edge (butuh 3-8 detik)...")
    print("-" * 76)

    # Jalankan tunnel
    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # Monitor output untuk menemukan URL
    try:
        while True:
            line = proc.stderr.readline()
            if not line and proc.poll() is not None:
                break

            if not tunnel_url:
                match = url_pattern.search(line)
                if match:
                    tunnel_url = match.group(0)
                    copied = copy_to_clipboard(tunnel_url)

                    print("\n" + "=" * 76)
                    print("  🎉 TUNNEL HTTPS BERHASIL DIBUAT & SIAP DIGUNAKAN!")
                    print("=" * 76)
                    print(f"\n  👉 BUKA LINK BERIKUT DI BROWSER HP ANDA (Chrome / Safari):")
                    print(f"     \033[1;32m{tunnel_url}\033[0m")
                    if copied:
                        print(f"\n  📋 [INFO] URL di atas telah OTOMATIS disalin ke Clipboard Anda!")
                        print(f"     (Tinggal tekan CTRL+V / Paste di WA / Telegram untuk dikirim ke HP)")
                    print("\n  💡 Petunjuk Pengujian:")
                    print("     1. Buka link HTTPS di atas pada HP.")
                    print("     2. Beri izin 'Allow' saat browser meminta akses Kamera.")
                    print("     3. Mulai uji coba sesi koin (kamera belakang aktif).")
                    print("     4. Jangan tutup jendela ini selama pengujian berlangsung!")
                    print("=" * 76 + "\n")
                    print("[LOG AKTIFITAS TUNNEL]:")
            else:
                # Tampilkan log ringkas jika ada koneksi
                if "200 OK" in line or "GET" in line or "POST" in line:
                    sys.stdout.write(f"  [HP -> Server] {line.strip()}\n")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[INFO] Menghentikan tunnel...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print("[INFO] Tunnel telah ditutup.")

if __name__ == "__main__":
    main()

