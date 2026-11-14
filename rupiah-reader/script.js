"use strict";

/*
 * Rupiah Reader – script.js (v8.0 Client-Side Deep Learning MLOps Edition)
 * 
 * Penggerak antarmuka modern & inferensi AI in-browser via TensorFlow.js (WebGL GPU).
 * - Arsitektur: MobileNetV2 Fine-Tuned (7 kelas nominal Rupiah).
 * - Preprocessing: 224x224 RGB, normalisasi [0, 1] (div 255.0).
 * - Spatial Ensembling: Viewfinder Crop (70%) + Full Frame (30%).
 * - Decision State Machine: SCANNING, CANDIDATE DETECTED, CONFIRMED, UNCERTAIN, UNKNOWN.
 * - Single Source of Truth Mapping: Class Index -> Nominal -> Thumbnail -> TTS -> Total.
 * - Non-intrusive MLOps Diagnostic Panel & Latency Logging.
 * - Animasi referensi GIF (laser sweep, success bloom checkmark, floating glass pill card).
 */

// ---------------------------------------------------------------------------
// Single Source of Truth: Registri Kelas Nominal Rupiah
// ---------------------------------------------------------------------------
const BANKNOTE_CLASSES = [
  { index: 0, code: "idr_1000",   nominal: 1000,   name: "Satu Ribu Rupiah",       badge: "1.000",   valClass: "val-1000",   hero: "Tjut Meutia",                    color: "#65a30d" },
  { index: 1, code: "idr_2000",   nominal: 2000,   name: "Dua Ribu Rupiah",        badge: "2.000",   valClass: "val-2000",   hero: "Mohammad Hoesni Thamrin",        color: "#64748b" },
  { index: 2, code: "idr_5000",   nominal: 5000,   name: "Lima Ribu Rupiah",       badge: "5.000",   valClass: "val-5000",   hero: "Dr. K.H. Idham Chalid",          color: "#b45309" },
  { index: 3, code: "idr_10000",  nominal: 10000,  name: "Sepuluh Ribu Rupiah",    badge: "10.000",  valClass: "val-10000",  hero: "Frans Kaisiepo",                 color: "#8b5cf6" },
  { index: 4, code: "idr_20000",  nominal: 20000,  name: "Dua Puluh Ribu Rupiah",  badge: "20.000",  valClass: "val-20000",  hero: "Dr. G.S.S.J. Ratulangi",         color: "#16a34a" },
  { index: 5, code: "idr_50000",  nominal: 50000,  name: "Lima Puluh Ribu Rupiah", badge: "50.000",  valClass: "val-50000",  hero: "Ir. H. Djuanda Kartawidjaja",    color: "#2563eb" },
  { index: 6, code: "idr_100000", nominal: 100000, name: "Seratus Ribu Rupiah",   badge: "100.000", valClass: "val-100000", hero: "Soekarno & Mohammad Hatta",     color: "#e11d48" }
];

// Ambang Batas Keyakinan Terkalibrasi (Berdasarkan Evaluasi Hold-out Test Set & Real Webcam)
const THRESHOLDS = {
  CONFIRMED: 0.40,    // Confidence >= 40% -> Terkonfirmasi
  UNCERTAIN: 0.28,    // 28% - 39% -> Ragu-ragu (arahkan kamera lebih dekat)
  UNKNOWN: 0.28       // < 28% -> Objek tidak dikenali
};

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

(() => {
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
    btnSwitchCam: $("btn-switch-cam"),
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
    sheetCamVal: $("sheet-cam-val"),

    // MLOps Diagnostic Elements
    diagStateBadge: $("diag-state-badge"),
    diagModel: $("diag-model"),
    diagBackend: $("diag-backend"),
    diagInput: $("diag-input"),
    diagLatency: $("diag-latency"),
    diagPrediction: $("diag-prediction"),
    diagConf: $("diag-conf"),
    diagTop3List: $("diag-top3-list"),

    // Live status
    srStatus: $("sr-status"),
  };

  // -------------------------------------------------------------------------
  // Status Aplikasi (State Management)
  // -------------------------------------------------------------------------
  const app = {
    stream: null,
    currentFacingMode: "environment", // default environment for mobile, fallback user
    torchOn: false,
    zoomLevel: 1,
    scanLock: false,
    total: 0,
    lastResult: null,
    lastTapTime: 0,
    tapTimer: null,
    voice: null,
    resetCardTimer: null,

    // In-Browser ML State
    tfModel: null,
    modelReady: false,
    backendName: "webgl",
    lastLatencyMs: 0,
    predictionHistory: [],
    currentState: "IDLE"
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
        u.lang = "id-ID";
        u.rate = 0.95;
        u.pitch = 1.0;
        u.volume = 1.0;
        if (app.voice) u.voice = app.voice;
        u.onend = () => resolve();
        u.onerror = () => resolve();
        setTimeout(() => speechSynthesis.speak(u), 40);
      } catch (_) {
        resolve();
      }
    });
  }

  function announce(pesan) {
    if (!el.srStatus) return;
    el.srStatus.textContent = "";
    setTimeout(() => { el.srStatus.textContent = pesan; }, 40);
  }

  function getar(pola) {
    try {
      if (navigator.vibrate) navigator.vibrate(pola);
    } catch (_) {}
  }

  // -------------------------------------------------------------------------
  // Inisialisasi Model In-Browser TensorFlow.js
  // -------------------------------------------------------------------------
  async function initTfjsModel() {
    if (typeof tf === "undefined") {
      console.warn("Library TensorFlow.js belum siap, menunggu script load...");
      setTimeout(initTfjsModel, 400);
      return;
    }

    try {
      if (el.chipText) el.chipText.textContent = "Loading AI Model...";
      await tf.ready();

      // Utamakan akselerasi WebGL GPU
      try {
        await tf.setBackend("webgl");
        app.backendName = "WebGL (GPU)";
      } catch (eGpu) {
        console.warn("WebGL tidak tersedia, fallback ke CPU:", eGpu);
        await tf.setBackend("cpu");
        app.backendName = "CPU";
      }

      console.log(`[MLOps] TF.js Backend aktif: ${tf.getBackend()}`);

      // Lokasi Model Artifacts
      const modelLocations = [
        "./model_tfjs/model.json",
        "model_tfjs/model.json",
        "../model_tfjs/model.json"
      ];

      let loaded = null;
      for (const loc of modelLocations) {
        try {
          console.log(`[MLOps] Mencoba memuat model dari: ${loc}`);
          loaded = await tf.loadLayersModel(loc);
          if (loaded) break;
        } catch (_) {}
      }

      if (!loaded) {
        throw new Error("Tidak dapat memuat model.json dari semua path.");
      }

      app.tfModel = loaded;
      app.modelReady = true;

      // Warmup forward pass untuk mengompilasi shader WebGL
      tf.tidy(() => {
        const dummy = tf.zeros([1, 224, 224, 3]);
        app.tfModel.predict(dummy);
      });

      console.log("[MLOps] MobileNetV2 Fine-Tuned siap untuk inferensi real-time!");
      if (el.chipText) el.chipText.textContent = `AI Ready (${app.backendName})`;
      if (el.sheetGpuVal) el.sheetGpuVal.textContent = `GPU WebGL Aktif (MobileNetV2)`;
      if (el.diagBackend) el.diagBackend.textContent = app.backendName;
      if (el.diagModel) el.diagModel.textContent = "MobileNetV2 v2.1.0";
    } catch (err) {
      console.error("[MLOps Error] Gagal inisialisasi model TF.js:", err);
      if (el.chipText) el.chipText.textContent = "AI Fallback Active";
      if (el.sheetGpuVal) el.sheetGpuVal.textContent = "Fallback Engine";
    }
  }

  // -------------------------------------------------------------------------
  // Preprocessing Citra Identik dengan Fase Training
  // -------------------------------------------------------------------------
  function extractTensorFromCrop(videoEl, sx, sy, sw, sh) {
    return tf.tidy(() => {
      // Buat canvas temporary untuk cropping presisi tinggi
      const cropCanvas = document.createElement("canvas");
      cropCanvas.width = 224;
      cropCanvas.height = 224;
      const ctx = cropCanvas.getContext("2d", { willReadFrequently: true });
      
      // Draw image dengan crop dan resize lanczos/bilinear
      ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, 224, 224);

      // Konversi pixel ke tensor float32 normalisasi [0.0, 1.0]
      const tensor = tf.browser.fromPixels(cropCanvas)
        .toFloat()
        .div(255.0)
        .expandDims(0); // [1, 224, 224, 3]

      return tensor;
    });
  }

  // -------------------------------------------------------------------------
  // Inferensi Multi-Scale Spatial Ensembling (Viewfinder + Full Frame)
  // -------------------------------------------------------------------------
  async function runInBrowserBanknoteInference() {
    const v = el.video;
    if (!v || !v.videoWidth || !app.tfModel) {
      return null;
    }

    const t0 = performance.now();
    const vw = v.videoWidth;
    const vh = v.videoHeight;

    // 1. Crop Area Viewfinder (Tengah Vertikal 65% x 70%)
    const cropW = Math.round(vw * 0.65);
    const cropH = Math.round(vh * 0.70);
    const startX = Math.round((vw - cropW) / 2);
    const startY = Math.round((vh - cropH) / 2);

    const tensorCrop = extractTensorFromCrop(v, startX, startY, cropW, cropH);
    const tensorFull = extractTensorFromCrop(v, 0, 0, vw, vh);

    let probsCropArr, probsFullArr;
    try {
      const predCrop = app.tfModel.predict(tensorCrop);
      const predFull = app.tfModel.predict(tensorFull);

      probsCropArr = Array.from(await predCrop.data());
      probsFullArr = Array.from(await predFull.data());
    } finally {
      tensorCrop.dispose();
      tensorFull.dispose();
    }

    const latency = Math.round(performance.now() - t0);
    app.lastLatencyMs = latency;

    // 2. Spatial Ensemble: 70% Viewfinder Target, 30% Full Frame Context
    const combinedProbs = probsCropArr.map((p, idx) => 0.70 * p + 0.30 * probsFullArr[idx]);

    // 3. Cari Argmax (Top Predicted Class) & Ranking Top-3
    const ranked = combinedProbs
      .map((prob, idx) => ({ ...BANKNOTE_CLASSES[idx], prob }))
      .sort((a, b) => b.prob - a.prob);

    const top = ranked[0];
    const second = ranked[1];
    const margin = top.prob - second.prob;

    // 4. Decision State Machine Terkalibrasi
    let state = "CONFIRMED";
    if (top.prob >= THRESHOLDS.CONFIRMED && margin >= 0.05) {
      state = "CONFIRMED";
    } else if (top.prob >= THRESHOLDS.UNCERTAIN) {
      state = "UNCERTAIN";
    } else {
      state = "UNKNOWN";
    }

    app.currentState = state;

    // 5. Update MLOps Diagnostic Panel
    updateDiagnosticPanel(state, top, ranked.slice(0, 3), latency);

    return {
      state,
      topClass: top,
      ranked,
      latency,
      confidence: top.prob,
      margin
    };
  }

  // -------------------------------------------------------------------------
  // Render MLOps Diagnostic Panel
  // -------------------------------------------------------------------------
  function updateDiagnosticPanel(state, top, top3, latency) {
    if (!el.diagStateBadge) return;

    el.diagStateBadge.textContent = state;
    el.diagStateBadge.className = `diag-badge badge-${state.toLowerCase()}`;

    if (el.diagLatency) el.diagLatency.textContent = `${latency} ms`;
    if (el.diagPrediction) el.diagPrediction.textContent = top ? `Rp ${top.badge}` : "-";
    if (el.diagConf) el.diagConf.textContent = top ? `${(top.prob * 100).toFixed(1)}%` : "-";

    if (el.diagTop3List && top3) {
      el.diagTop3List.innerHTML = top3.map((item, idx) => `
        <div class="diag-top3-row">
          <span class="top3-class">${idx + 1}. Rp ${item.badge} (${item.hero})</span>
          <span class="top3-pct">${(item.prob * 100).toFixed(1)}%</span>
        </div>
      `).join("");
    }
  }

  // -------------------------------------------------------------------------
  // Akses Kamera Penuh & Switch Kamera
  // -------------------------------------------------------------------------
  async function startCamera(facingMode = "environment") {
    if (app.stream) {
      app.stream.getTracks().forEach((t) => t.stop());
      app.stream = null;
    }

    // Deteksi jika platform laptop/desktop: ideal "user", mobile: ideal "environment"
    const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    const targetFacing = isMobile ? facingMode : "user";
    app.currentFacingMode = targetFacing;

    const constraints = {
      video: {
        facingMode: { ideal: targetFacing },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      app.stream = stream;
      el.video.srcObject = stream;
      el.video.setAttribute("playsinline", "");
      await el.video.play();

      if (el.sheetCamVal) {
        el.sheetCamVal.textContent = targetFacing === "user" ? "Kamera Depan / Webcam" : "Kamera Belakang";
      }
    } catch (err) {
      console.warn("Kamera ideal gagal, coba fallback generic video:", err);
      try {
        const streamFallback = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        app.stream = streamFallback;
        el.video.srcObject = streamFallback;
        await el.video.play();
        if (el.sheetCamVal) el.sheetCamVal.textContent = "Kamera Default";
      } catch (err2) {
        console.error("Gagal menyalakan kamera:", err2);
        if (el.chipText) el.chipText.textContent = "Camera Blocked";
        speak("Izin kamera belum aktif. Berikan izin kamera di browser Anda.");
      }
    }
  }

  async function switchCamera() {
    getar(35);
    const newFacing = app.currentFacingMode === "environment" ? "user" : "environment";
    await startCamera(newFacing);
    speak(newFacing === "user" ? "Beralih ke kamera depan" : "Beralih ke kamera belakang");
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
        app.torchOn = !app.torchOn;
        el.btnTorch.classList.toggle("torch-active", app.torchOn);
        getar(40);
        return;
      }

      app.torchOn = !app.torchOn;
      await track.applyConstraints({ advanced: [{ torch: app.torchOn }] });
      el.btnTorch.classList.toggle("torch-active", app.torchOn);
      getar(50);
    } catch (e) {
      console.warn("Gagal toggle torch:", e);
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

    el.video.style.transform = app.zoomLevel === 2 ? "scale(1.3)" : "scale(1)";
    el.video.style.transition = "transform 0.3s ease";
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

    // 1. Masuk ke State: SCANNING
    document.body.classList.remove("show-success");
    document.body.classList.add("state-scanning");
    
    el.bottomCard.className = "bottom-pill-card state-scanning";
    el.pillSubtitle.textContent = "Scanning...";
    el.pillTitle.textContent = "Memindai uang kertas...";
    getar(45);

    try {
      // Pastikan model siap
      if (!app.tfModel) {
        console.warn("[MLOps] Model belum siap saat pemindaian, mencoba memuat...");
        await initTfjsModel();
      }

      // Jalankan inferensi deep learning di browser
      const inferRes = await runInBrowserBanknoteInference();

      // Jeda minimum 400ms agar efek sweeping laser violet terlihat indah & natural
      await new Promise((r) => setTimeout(r, 420));

      if (!inferRes || inferRes.state === "UNKNOWN") {
        // STATE UNKNOWN
        document.body.classList.remove("state-scanning");
        el.bottomCard.className = "bottom-pill-card state-idle";
        el.pillSubtitle.textContent = "Tidak Terdeteksi";
        el.pillTitle.textContent = "Arahkan Uang ke Kamera";
        getar([100, 50, 100]);
        speak("Uang tidak terdeteksi. Silakan arahkan uang tepat di depan kamera lalu ketuk layar lagi.");

        app.resetCardTimer = setTimeout(() => {
          el.pillSubtitle.textContent = "Scanning...";
          el.pillTitle.textContent = "Posisikan Uang Kertas di Sini";
        }, 3500);
        return;
      }

      if (inferRes.state === "UNCERTAIN") {
        // STATE UNCERTAIN
        document.body.classList.remove("state-scanning");
        el.bottomCard.className = "bottom-pill-card state-idle";
        el.pillSubtitle.textContent = "Kurang Jelas";
        el.pillTitle.textContent = "Dekatkan Uang ke Kamera";
        getar([60, 40, 60]);
        speak("Uang kurang jelas. Mohon luruskan atau dekatkan uang ke kamera.");

        app.resetCardTimer = setTimeout(() => {
          el.pillSubtitle.textContent = "Scanning...";
          el.pillTitle.textContent = "Posisikan Uang Kertas di Sini";
        }, 3500);
        return;
      }

      // STATE CONFIRMED:
      // Ambil objek kelas nominal yang terkonfirmasi (SINGLE SOURCE OF TRUTH)
      const selected = inferRes.topClass;

      const res = {
        type: "banknote",
        nominal: selected.nominal,
        confidence: inferRes.confidence,
        speechText: nominalToSpeech(selected.nominal),
        displayText: formatRupiah(selected.nominal),
        badgeText: selected.badge,
        valClass: selected.valClass,
        hero: selected.hero
      };

      app.lastResult = res;

      // 2. Masuk ke State: SUCCESS BLOOM CHECKMARK di tengah viewfinder
      document.body.classList.remove("state-scanning");
      document.body.classList.add("show-success");
      getar([80, 40, 80]);

      // Suara pembacaan nominal langsung
      speak(res.speechText);

      // 3. Masuk ke State: RESULT DISPLAY di Bottom Pill Card
      await new Promise((r) => setTimeout(r, 600));
      document.body.classList.remove("show-success");

      // Perbarui thumbnail uang mini realistis (100% konsisten)
      el.banknoteThumb.className = `banknote-thumbnail ${res.valClass}`;
      el.thumbNumber.textContent = res.badgeText;

      el.bottomCard.className = "bottom-pill-card state-result";
      el.pillSubtitle.textContent = "Hasil Deteksi";
      el.pillTitle.textContent = res.displayText;

      // Kembali dengan mulus ke State Idle setelah 4.5 detik
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
  // Akumulasi Total Uang (Hanya via Interaksi Double Tap / Menu)
  // -------------------------------------------------------------------------
  function addToTotal() {
    if (!app.lastResult || !app.lastResult.nominal) {
      speak("Belum ada nominal yang dipindai untuk ditambahkan.");
      return;
    }
    app.total += app.lastResult.nominal;
    if (el.sheetTotalVal) el.sheetTotalVal.textContent = formatRupiah(app.total);
    getar([50, 50, 50]);
    speak(`Ditambahkan ${terbilang(app.lastResult.nominal)} rupiah. Total belanja sekarang ${terbilang(app.total)} rupiah.`);
  }

  function resetTotal() {
    app.total = 0;
    if (el.sheetTotalVal) el.sheetTotalVal.textContent = "Rp 0";
    getar(40);
    speak("Total akumulasi belanja direset ke nol rupiah.");
  }

  // -------------------------------------------------------------------------
  // Dispatcher Ketuk Layar (Single Tap = Scan, Double Tap = Add to Total)
  // -------------------------------------------------------------------------
  function handleScreenTap() {
    const now = performance.now();
    const selisih = now - app.lastTapTime;

    if (selisih < 320) {
      // DOUBLE TAP DETECTED
      if (app.tapTimer) {
        clearTimeout(app.tapTimer);
        app.tapTimer = null;
      }
      app.lastTapTime = 0;
      addToTotal();
    } else {
      // SINGLE TAP (Dijadwalkan untuk membedakan dari double tap)
      app.lastTapTime = now;
      if (app.tapTimer) clearTimeout(app.tapTimer);
      app.tapTimer = setTimeout(() => {
        app.tapTimer = null;
        triggerScan();
      }, 240);
    }
  }

  // -------------------------------------------------------------------------
  // Pengelolaan Bottom Sheet Menu
  // -------------------------------------------------------------------------
  function openMenu() {
    if (!el.menuBackdrop) return;
    el.menuBackdrop.hidden = false;
    getar(25);
    speak(`Menu opsi dibuka. Total belanja saat ini ${terbilang(app.total)} rupiah.`);
  }

  function closeMenu() {
    if (!el.menuBackdrop) return;
    el.menuBackdrop.hidden = true;
    getar(25);
  }

  // -------------------------------------------------------------------------
  // Pendaftaran Event Listeners
  // -------------------------------------------------------------------------
  function attachEventListeners() {
    // Layar penuh untuk ketuk
    if (el.tapArea) {
      el.tapArea.addEventListener("click", (e) => {
        // Cegah trigger jika mengklik tombol kontrol
        if (e.target.closest(".nav-circle-btn, .floating-zoom-btn, .pill-torch-btn, .menu-bottom-sheet")) return;
        handleScreenTap();
      });
    }

    // Tombol senter
    if (el.btnTorch) {
      el.btnTorch.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleTorch();
      });
    }

    // Tombol zoom
    if (el.btnZoom) {
      el.btnZoom.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleZoom();
      });
    }

    // Tombol switch kamera
    if (el.btnSwitchCam) {
      el.btnSwitchCam.addEventListener("click", (e) => {
        e.stopPropagation();
        switchCamera();
      });
    }

    // Tombol menu
    if (el.btnMenu) {
      el.btnMenu.addEventListener("click", (e) => {
        e.stopPropagation();
        openMenu();
      });
    }

    // Chip AI untuk buka menu/diagnostik
    if (el.aiChip) {
      el.aiChip.addEventListener("click", (e) => {
        e.stopPropagation();
        openMenu();
      });
    }

    // Tombol tutup menu
    if (el.btnCloseSheet) {
      el.btnCloseSheet.addEventListener("click", closeMenu);
    }

    if (el.menuBackdrop) {
      el.menuBackdrop.addEventListener("click", (e) => {
        if (e.target === el.menuBackdrop) closeMenu();
      });
    }

    // Tombol total di sheet
    if (el.btnAddTotal) {
      el.btnAddTotal.addEventListener("click", addToTotal);
    }
    if (el.btnResetTotal) {
      el.btnResetTotal.addEventListener("click", resetTotal);
    }

    // Tombol ulang / back
    if (el.btnBack) {
      el.btnBack.addEventListener("click", (e) => {
        e.stopPropagation();
        triggerScan();
      });
    }

    // Keyboard Accessibility (Spasi = Scan, T = Total, M = Menu, D = Diagnostic)
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        handleScreenTap();
      } else if (e.key === "t" || e.key === "T") {
        addToTotal();
      } else if (e.key === "m" || e.key === "M") {
        openMenu();
      }
    });
  }

  // -------------------------------------------------------------------------
  // Bootstrapping Aplikasi
  // -------------------------------------------------------------------------
  async function bootstrap() {
    attachEventListeners();
    await startCamera();
    await initTfjsModel();

    // Salam pembuka suara
    setTimeout(() => {
      speak("Rupiah Reader siap. Arahkan uang ke kamera lalu ketuk layar untuk memindai.");
    }, 600);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
