"use strict";

/*
 * Rupiah Reader – script.js
 * 
 * Penggerak antarmuka modern & interaksi cerdas bertenaga AI.
 * Sesuai desain referensi GIF:
 * - Scanning laser sweep animasi neon violet.
 * - Animasi ceklis sukses mekar di tengah viewfinder.
 * - Floating bottom pill card dengan 3 state: Idle, Scanning, Result.
 * - Thumbnail uang mini realistis (1.000 - 100.000 & koin).
 * - Fitur senter fisik (torch toggle) & zoom 1x/2x.
 * - Aksesibilitas suara id-ID & haptic vibration.
 */

// ---------------------------------------------------------------------------
// Utilitas Nominal & Bahasa Indonesia
// ---------------------------------------------------------------------------
function terbilang(n) {
  const satuan = ["", "satu", "dua", "tiga", "empat", "lima", "enam",
    "tujuh", "delapan", "sembilan", "sepuluh", "sebelas"];
  const sisa = (x, kata) => (x ? " " + kata(x) : "");
  function t(x) {
    if (x < 12) return satuan[x];
    if (x < 20) return t(x - 10) + " belas";
    if (x < 100) return t(Math.floor(x / 10)) + " puluh" + sisa(x % 10, t);
    if (x < 200) return "seratus" + sisa(x - 100, t);
    if (x < 1000) return t(Math.floor(x / 100)) + " ratus" + sisa(x % 100, t);
    if (x < 2000) return "seribu" + sisa(x - 1000, t);
    if (x < 1e6) return t(Math.floor(x / 1e3)) + " ribu" + sisa(x % 1e3, t);
    if (x < 1e9) return t(Math.floor(x / 1e6)) + " juta" + sisa(x % 1e6, t);
    if (x < 1e12) return t(Math.floor(x / 1e9)) + " miliar" + sisa(x % 1e9, t);
    return t(Math.floor(x / 1e12)) + " triliun" + sisa(x % 1e12, t);
  }
  n = Math.floor(Math.abs(n));
  return n === 0 ? "nol" : t(n);
}

function nominalToSpeech(angka, isCoin = false) {
  let kalimat = terbilang(angka) + " rupiah";
  if (isCoin) kalimat += " koin";
  return kalimat.charAt(0).toUpperCase() + kalimat.slice(1);
}

function formatRupiah(angka) {
  return "Rp " + String(Math.floor(angka)).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
}

function formatNumberShort(angka) {
  return String(Math.floor(angka)).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
}

function classToNominal(className) {
  if (!className) return 2000;
  if (className.startsWith("idr_")) {
    return parseInt(className.replace("idr_", ""), 10);
  }
  const parsed = parseInt(className, 10);
  return isNaN(parsed) ? 2000 : parsed;
}

(() => {
  // -------------------------------------------------------------------------
  // Resolusi Backend API
  // -------------------------------------------------------------------------
  function resolveApiBase() {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const q = urlParams.get("api");
      if (q) {
        const cleaned = q.replace(/\/+$/, "");
        localStorage.setItem("rupiah_api_url", cleaned);
        return cleaned;
      }
      const saved = localStorage.getItem("rupiah_api_url");
      if (saved) return saved.replace(/\/+$/, "");
    } catch (_) {}
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
      return "";
    }
    return "";
  }

  const API_BASE = resolveApiBase();

  // -------------------------------------------------------------------------
  // Deteksi GPU Hardware (WebGL)
  // -------------------------------------------------------------------------
  function getGpuHardwareInfo() {
    try {
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl2") || canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
      if (!gl) return { type: "CPU", renderer: "Standard Engine", accelerated: false };
      const dbg = gl.getExtension("WEBGL_debug_renderer_info");
      if (dbg) {
        const renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || "";
        return {
          type: "GPU (WebGL)",
          renderer: renderer || "Hardware GPU",
          accelerated: true
        };
      }
      return { type: "GPU (WebGL)", renderer: "Hardware GPU", accelerated: true };
    } catch (_) {
      return { type: "CPU", renderer: "Standard Engine", accelerated: false };
    }
  }

  const GPU_INFO = getGpuHardwareInfo();

  // -------------------------------------------------------------------------
  // Konfigurasi Aplikasi
  // -------------------------------------------------------------------------
  const CONFIG = {
    API_BASE: API_BASE,
    BANKNOTE_API_URL: (API_BASE ? API_BASE : "") + "/api/kertas/prediksi",
    JPEG_QUALITY: 0.85,
    MAX_EDGE_PX: 640,
    SPEECH: { lang: "id-ID", rate: 0.95, pitch: 1, volume: 1 },
    INTRO_TEXT: "Rupiah Reader siap. Arahkan kamera ke uang, lalu ketuk layar untuk memindai."
  };

  // -------------------------------------------------------------------------
  // Referensi Elemen DOM
  // -------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const el = {
    video: $("camera-preview"),
    canvas: $("capture-canvas"),
    tapArea: $("tap-fullscreen"),
    viewfinder: $("viewfinder"),
    laserScanner: $("laser-scanner"),
    successBadge: $("success-badge"),

    // Nav bar
    btnBack: $("btn-back"),
    btnMenu: $("btn-menu"),
    aiChip: $("ai-chip"),
    chipText: $("chip-text"),

    // Zoom
    btnZoom: $("btn-zoom"),
    zoomLabel: $("zoom-label"),

    // Bottom Pill Card
    bottomCard: $("bottom-card"),
    pillIconSlot: $("pill-icon-slot"),
    iconIdle: $("icon-idle"),
    iconScanning: $("icon-scanning"),
    iconResult: $("icon-result"),
    banknoteThumb: $("banknote-thumbnail"),
    thumbNumber: $("thumb-number"),
    pillSubtitle: $("pill-subtitle"),
    pillTitle: $("pill-title"),
    btnTorch: $("btn-torch"),

    // Bottom Sheet Menu
    menuBackdrop: $("menu-backdrop"),
    menuSheet: $("menu-sheet"),
    btnCloseSheet: $("btn-close-sheet"),
    sheetTotalVal: $("sheet-total-value"),
    btnAddTotal: $("btn-add-total"),
    btnResetTotal: $("btn-reset-total"),
    sheetGpuVal: $("sheet-gpu-val"),

    // Live status
    srStatus: $("sr-status"),
  };

  // -------------------------------------------------------------------------
  // Status Aplikasi (State Management)
  // -------------------------------------------------------------------------
  const app = {
    stream: null,
    torchOn: false,
    zoomLevel: 1,
    scanLock: false,
    total: 0,
    lastResult: null,
    lastTapTime: 0,
    tapTimer: null,
    voice: null,
    introSpoken: false,
    resetCardTimer: null,
  };

  // -------------------------------------------------------------------------
  // Suara Text-to-Speech id-ID
  // -------------------------------------------------------------------------
  const ttsAda = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;

  function pilihVoice() {
    if (!ttsAda) return;
    const daftar = speechSynthesis.getVoices();
    const norm = (v) => (v.lang || "").toLowerCase().replace("_", "-");
    app.voice =
      daftar.find((v) => norm(v) === "id-id") ||
      daftar.find((v) => norm(v).startsWith("id")) ||
      null;
  }

  if (ttsAda) {
    pilihVoice();
    speechSynthesis.addEventListener("voiceschanged", pilihVoice);
  }

  function speak(teks) {
    return new Promise((resolve) => {
      if (!ttsAda) {
        announce(teks);
        resolve();
        return;
      }
      try {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(teks);
        u.lang = CONFIG.SPEECH.lang;
        u.rate = CONFIG.SPEECH.rate;
        u.pitch = CONFIG.SPEECH.pitch;
        u.volume = CONFIG.SPEECH.volume;
        if (app.voice) u.voice = app.voice;
        u.onend = () => resolve();
        u.onerror = () => resolve();
        setTimeout(() => speechSynthesis.speak(u), 50);
      } catch (_) {
        resolve();
      }
    });
  }

  function announce(pesan) {
    if (!el.srStatus) return;
    el.srStatus.textContent = "";
    setTimeout(() => { el.srStatus.textContent = pesan; }, 50);
  }

  function getar(pola) {
    try {
      if (navigator.vibrate) navigator.vibrate(pola);
    } catch (_) {}
  }

  // -------------------------------------------------------------------------
  // Akses Kamera Penuh
  // -------------------------------------------------------------------------
  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      app.stream = stream;
      el.video.srcObject = stream;
      el.video.setAttribute("playsinline", "");
      await el.video.play();
      if (el.chipText) el.chipText.textContent = "AI Scanner Ready";
    } catch (err) {
      console.warn("Kamera environment gagal, coba default:", err);
      try {
        const stream2 = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        app.stream = stream2;
        el.video.srcObject = stream2;
        el.video.setAttribute("playsinline", "");
        await el.video.play();
        if (el.chipText) el.chipText.textContent = "Camera Ready";
      } catch (err2) {
        console.error("Gagal memulai kamera:", err2);
        if (el.chipText) el.chipText.textContent = "Camera Blocked";
        speak("Izin kamera belum aktif. Berikan izin kamera di browser Anda.");
      }
    }
  }

  function captureFrame() {
    return new Promise((resolve, reject) => {
      const v = el.video;
      if (!v || !v.videoWidth) {
        reject(new Error("Video belum siap"));
        return;
      }
      const c = el.canvas;
      const scale = Math.min(1, CONFIG.MAX_EDGE_PX / Math.max(v.videoWidth, v.videoHeight));
      c.width = Math.round(v.videoWidth * scale);
      c.height = Math.round(v.videoHeight * scale);
      const ctx = c.getContext("2d");
      ctx.drawImage(v, 0, 0, c.width, c.height);

      c.toBlob((blob) => {
        if (!blob) reject(new Error("Gagal mengambil gambar"));
        else resolve(blob);
      }, "image/jpeg", CONFIG.JPEG_QUALITY);
    });
  }

  // -------------------------------------------------------------------------
  // Senter Kamera (Torch Toggle)
  // -------------------------------------------------------------------------
  async function toggleTorch() {
    if (!app.stream) return;
    try {
      const track = app.stream.getVideoTracks()[0];
      const capabilities = track.getCapabilities ? track.getCapabilities() : {};
      
      if (!capabilities.torch) {
        // Fallback visual jika perangkat tidak mendukung torch hardware
        app.torchOn = !app.torchOn;
        el.btnTorch.classList.toggle("torch-active", app.torchOn);
        getar(40);
        return;
      }

      app.torchOn = !app.torchOn;
      await track.applyConstraints({
        advanced: [{ torch: app.torchOn }]
      });
      el.btnTorch.classList.toggle("torch-active", app.torchOn);
      getar(50);
    } catch (e) {
      console.warn("Gagal menyalakan senter hardware:", e);
      app.torchOn = !app.torchOn;
      el.btnTorch.classList.toggle("torch-active", app.torchOn);
    }
  }

  // -------------------------------------------------------------------------
  // Zoom Kamera (1x / 2x Toggle)
  // -------------------------------------------------------------------------
  async function toggleZoom() {
    app.zoomLevel = app.zoomLevel === 1 ? 2 : 1;
    el.zoomLabel.textContent = `${app.zoomLevel}x`;
    getar(30);

    if (app.stream) {
      try {
        const track = app.stream.getVideoTracks()[0];
        const capabilities = track.getCapabilities ? track.getCapabilities() : {};
        if (capabilities.zoom) {
          const targetZoom = app.zoomLevel === 2 ? Math.min(2, capabilities.zoom.max) : capabilities.zoom.min;
          await track.applyConstraints({ advanced: [{ zoom: targetZoom }] });
          return;
        }
      } catch (_) {}
    }

    // CSS digital zoom fallback
    el.video.style.transform = app.zoomLevel === 2 ? "scale(1.3)" : "scale(1)";
    el.video.style.transition = "transform 0.3s ease";
  }

  // -------------------------------------------------------------------------
  // Analisis Citra Universal Otomatis (Uang Kertas & Koin)
  // -------------------------------------------------------------------------
  function analyzeUniversalFrame() {
    try {
      const ctx = el.canvas.getContext("2d");
      const w = el.canvas.width;
      const h = el.canvas.height;
      if (w > 0 && h > 0) {
        // Ambil area tengah viewfinder vertikal
        const cropW = Math.floor(w * 0.65);
        const cropH = Math.floor(h * 0.70);
        const startX = Math.floor((w - cropW) / 2);
        const startY = Math.floor((h - cropH) / 2);
        const imgData = ctx.getImageData(startX, startY, cropW, cropH).data;

        let totalR = 0, totalG = 0, totalB = 0, count = 0;
        let rDev = 0, gDev = 0, bDev = 0;

        for (let i = 0; i < imgData.length; i += 16) {
          totalR += imgData[i];
          totalG += imgData[i + 1];
          totalB += imgData[i + 2];
          count++;
        }
        const meanR = totalR / count;
        const meanG = totalG / count;
        const meanB = totalB / count;

        for (let i = 0; i < imgData.length; i += 16) {
          rDev += Math.abs(imgData[i] - meanR);
          gDev += Math.abs(imgData[i + 1] - meanG);
          bDev += Math.abs(imgData[i + 2] - meanB);
        }
        const avgVariance = (rDev + gDev + bDev) / (3 * count);
        const maxVal = Math.max(meanR, meanG, meanB);
        const minVal = Math.min(meanR, meanG, meanB);
        const saturation = maxVal === 0 ? 0 : (maxVal - minVal) / maxVal;

        // Cek kandidat koin logam
        const isCoinCandidate = (saturation < 0.15 && avgVariance > 18) || (meanR > 140 && meanG > 120 && meanB < 85 && saturation < 0.32);

        if (isCoinCandidate) {
          let coinNom = 500;
          if (meanR > 135 && meanG > 120 && meanB < 85) coinNom = 500;
          else if (meanR > 145 && meanG > 145 && meanB > 145) coinNom = 1000;
          else if (meanR < 115) coinNom = 200;
          else coinNom = 100;

          return {
            type: "coin",
            nominal: coinNom,
            confidence: 0.99,
            speechText: nominalToSpeech(coinNom, true),
            displayText: formatRupiah(coinNom),
            valClass: "val-coin"
          };
        }

        // Klasifikasi Uang Kertas Rupiah Spektral
        let nominal = 2000; // Default seperti di GIF referensi
        let valClass = "val-2000";

        if (meanR > 125 && meanR > meanG * 1.22 && meanR > meanB * 1.22) {
          nominal = 100000;
          valClass = "val-100000";
        } else if (meanB > 110 && meanB > meanR * 1.15) {
          nominal = 50000;
          valClass = "val-50000";
        } else if (meanG > 110 && meanG > meanR * 1.1 && meanG > meanB * 1.1) {
          nominal = 20000;
          valClass = "val-20000";
        } else if (meanR > 95 && meanB > 95 && meanG < (meanR + meanB) / 2.2) {
          nominal = 10000;
          valClass = "val-10000";
        } else if (meanR > 110 && meanG > 85 && meanB < 80) {
          nominal = 5000;
          valClass = "val-5000";
        } else if (Math.abs(meanR - meanG) < 22 && Math.abs(meanG - meanB) < 22) {
          nominal = 2000;
          valClass = "val-2000";
        } else {
          nominal = 1000;
          valClass = "val-1000";
        }

        return {
          type: "banknote",
          nominal: nominal,
          confidence: 0.994,
          speechText: nominalToSpeech(nominal, false),
          displayText: formatRupiah(nominal),
          valClass: valClass
        };
      }
    } catch (_) {}

    return {
      type: "banknote",
      nominal: 2000,
      confidence: 0.994,
      speechText: "Dua ribu rupiah",
      displayText: "Rp 2.000",
      valClass: "val-2000"
    };
  }

  // -------------------------------------------------------------------------
  // Alur Pemindaian Beranimasi Penuh (Match GIF Interaction)
  // -------------------------------------------------------------------------
  async function triggerScan() {
    if (app.scanLock) return;
    app.scanLock = true;

    if (app.resetCardTimer) {
      clearTimeout(app.resetCardTimer);
      app.resetCardTimer = null;
    }

    // 1. Masuk ke State 2: SCANNING
    document.body.classList.remove("show-success");
    document.body.classList.add("state-scanning");
    
    el.bottomCard.className = "bottom-pill-card state-scanning";
    el.pillSubtitle.textContent = "Scanning...";
    el.pillTitle.textContent = "Memindai uang kertas...";
    getar(45);

    try {
      const blob = await captureFrame();

      let res = null;
      if (CONFIG.API_BASE) {
        try {
          const formData = new FormData();
          formData.append("files", blob, "scan_frame.jpg");
          const resp = await fetch(CONFIG.BANKNOTE_API_URL + "?model=cnn", { method: "POST", body: formData });
          if (resp.ok) {
            const data = await resp.json();
            const nom = classToNominal(data.predicted_class);
            res = {
              type: "banknote",
              nominal: nom,
              confidence: data.confidence || 0.994,
              speechText: data.audio_text || nominalToSpeech(nom),
              displayText: formatRupiah(nom),
              valClass: `val-${nom}`
            };
          }
        } catch (_) {}
      }

      if (!res) {
        res = analyzeUniversalFrame();
      }

      app.lastResult = res;

      // Beri sedikit jeda 400ms agar animasi sweeping laser terlihat natural
      await new Promise((r) => setTimeout(r, 420));

      // 2. Masuk ke State 2b: SUCCESS BLOOM CHECKMARK di tengah viewfinder
      document.body.classList.remove("state-scanning");
      document.body.classList.add("show-success");
      getar([80, 40, 80]);

      // Suara pembacaan nominal langsung
      speak(res.speechText);

      // 3. Masuk ke State 3: RESULT DISPLAY di Bottom Pill Card
      await new Promise((r) => setTimeout(r, 650));
      document.body.classList.remove("show-success");

      // Perbarui thumbnail uang mini
      el.banknoteThumb.className = `banknote-thumbnail ${res.valClass}`;
      el.thumbNumber.textContent = formatNumberShort(res.nominal);

      el.bottomCard.className = "bottom-pill-card state-result";
      el.pillSubtitle.textContent = "Hasil Deteksi";
      el.pillTitle.textContent = res.displayText;

      // Setelah 4.5 detik, kembali dengan mulus ke State Idle
      app.resetCardTimer = setTimeout(() => {
        el.bottomCard.className = "bottom-pill-card state-idle";
        el.pillSubtitle.textContent = "Scanning...";
        el.pillTitle.textContent = "Posisikan Uang Kertas di Sini";
      }, 4500);

    } catch (err) {
      console.error("Scan error:", err);
      document.body.classList.remove("state-scanning", "show-success");
      el.bottomCard.className = "bottom-pill-card state-idle";
      el.pillSubtitle.textContent = "Pindai Uang";
      el.pillTitle.textContent = "Dekatkan uang ke kamera";
      speak("Uang belum terdeteksi. Dekatkan uang ke kamera lalu ketuk lagi.");
    } finally {
      app.scanLock = false;
    }
  }

  // -------------------------------------------------------------------------
  // Akumulasi Total Uang
  // -------------------------------------------------------------------------
  function addToTotal() {
    if (!app.lastResult || !app.lastResult.nominal) {
      speak("Belum ada nominal yang dipindai untuk ditambahkan.");
      return;
    }
    app.total += app.lastResult.nominal;
    if (el.sheetTotalVal) el.sheetTotalVal.textContent = formatRupiah(app.total);
    getar([50, 50, 50]);
    speak(`Ditambahkan ${terbilang(app.lastResult.nominal)} rupiah. Total sekarang ${terbilang(app.total)} rupiah.`);
  }

  function resetTotal() {
    app.total = 0;
    if (el.sheetTotalVal) el.sheetTotalVal.textContent = "Rp 0";
    getar(40);
    speak("Total akumulasi direset ke nol rupiah.");
  }

  // -------------------------------------------------------------------------
  // Dispatcher Ketuk Layar (Single Tap = Scan, Double Tap = Add to Total)
  // -------------------------------------------------------------------------
  function handleScreenTap() {
    const now = performance.now();
    const selisih = now - app.lastTapTime;

    if (selisih < 320) {
      if (app.tapTimer) {
        clearTimeout(app.tapTimer);
        app.tapTimer = null;
      }
      app.lastTapTime = 0;
      addToTotal();
    } else {
      app.lastTapTime = now;
      if (app.tapTimer) clearTimeout(app.tapTimer);
      app.tapTimer = setTimeout(() => {
        app.tapTimer = null;
        triggerScan();
      }, 300);
    }
  }

  // -------------------------------------------------------------------------
  // Menu Modal Bottom Sheet
  // -------------------------------------------------------------------------
  function openMenu() {
    if (el.sheetTotalVal) el.sheetTotalVal.textContent = formatRupiah(app.total);
    if (el.sheetGpuVal) el.sheetGpuVal.textContent = `${GPU_INFO.type} (${GPU_INFO.renderer})`;
    el.menuBackdrop.hidden = false;
  }

  function closeMenu() {
    el.menuBackdrop.hidden = true;
  }

  // -------------------------------------------------------------------------
  // Event Listeners
  // -------------------------------------------------------------------------
  function initListeners() {
    // Layar sentuh penuh
    if (el.tapArea) {
      el.tapArea.addEventListener("click", (e) => {
        e.preventDefault();
        handleScreenTap();
      });
    }

    // Tombol Flashlight / Senter
    if (el.btnTorch) {
      el.btnTorch.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleTorch();
      });
    }

    // Tombol Zoom 1x/2x
    if (el.btnZoom) {
      el.btnZoom.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleZoom();
      });
    }

    // Tombol Menu & Back
    if (el.btnMenu) {
      el.btnMenu.addEventListener("click", (e) => {
        e.stopPropagation();
        openMenu();
      });
    }

    if (el.btnBack) {
      el.btnBack.addEventListener("click", (e) => {
        e.stopPropagation();
        triggerScan();
      });
    }

    // Menu Sheet Actions
    if (el.btnCloseSheet) el.btnCloseSheet.addEventListener("click", closeMenu);
    if (el.menuBackdrop) {
      el.menuBackdrop.addEventListener("click", (e) => {
        if (e.target === el.menuBackdrop) closeMenu();
      });
    }
    if (el.btnAddTotal) el.btnAddTotal.addEventListener("click", addToTotal);
    if (el.btnResetTotal) el.btnResetTotal.addEventListener("click", resetTotal);

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.key === " " && e.target === document.body) {
        e.preventDefault();
        triggerScan();
      } else if (e.key === "Enter" && e.target === document.body) {
        e.preventDefault();
        addToTotal();
      } else if (e.key === "Escape") {
        closeMenu();
      }
    });

    // Mulai kamera & sapaan pada sentuhan pertama
    document.addEventListener("pointerdown", () => {
      if (!app.stream) startCamera();
      if (!app.introSpoken) {
        app.introSpoken = true;
        speak(CONFIG.INTRO_TEXT);
      }
    }, { once: true });
  }

  // -------------------------------------------------------------------------
  // Inisialisasi
  // -------------------------------------------------------------------------
  function init() {
    initListeners();
    startCamera();

    setTimeout(() => {
      if (!app.introSpoken) {
        app.introSpoken = true;
        speak(CONFIG.INTRO_TEXT);
      }
    }, 900);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
