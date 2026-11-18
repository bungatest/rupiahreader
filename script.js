(() => {
  "use strict";

  /*
   * Rupiah Reader – script.js (v10.1 Multi-Crop Full-Frame � No Viewfinder Dependency)
   * 
   * Penggerak antarmuka modern & inferensi AI in-browser via TensorFlow.js (WebGL GPU).
   * - Seluruh area kamera adalah area deteksi (Bebas ketergantungan pada crop viewfinder statis).
   * - Lokalisasi Uang Otomatis (Client-side, Fast & Lightweight):
   *   Downscale ke 320px, analisis gradient/saliency, skin-tone filtering, bounding box dengan safety margin 12%.
   * - Multi-Crop Fallback (Candidate Crop + Full-Frame Letterbox Mode 'pad' 224x224 RGB [0, 255]).
   * - Continuous Autonomous Scanning (~5-7 FPS) dengan 6-Frame Sliding Window Consensus.
   * - Non-Spamming Audio Guidance System untuk Penyandang Tunanetra (Cooldown >= 3.8s).
   * - Visual Debug Mode (?debug=1 di URL): Bounding box overlay, PIP tensor preview 224x224, FPS, voting buffer.
   * - Single Source of Truth Class Mapping (idr_1000 .. idr_100000).
   * - Kompatibilitas penuh dengan coin-reader.js dan fitur koin.
   */

  // ---------------------------------------------------------------------------
  // Single Source of Truth: Registri Kelas Nominal Rupiah (Sesuai Urutan Notebook)
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
  window.BANKNOTE_CLASSES = BANKNOTE_CLASSES;

  // ---------------------------------------------------------------------------
  // Registri Kelas Nominal Uang Logam (Koin) Rupiah 5-Class (MobileNetV2 Fine-Tuned)
  // ---------------------------------------------------------------------------
  const COIN_CLASSES = [
    { index: 0, code: "100", nominal: 100, name: "Seratus Rupiah Koin", badge: "100", valClass: "val-coin-100", hero: "Prof. Herman Johannes", color: "#94a3b8" },
    { index: 1, code: "1000", nominal: 1000, name: "Seribu Rupiah Koin", badge: "1.000", valClass: "val-coin-1000", hero: "Mr. I Gusti Ketut Pudja", color: "#eab308" },
    { index: 2, code: "200", nominal: 200, name: "Dua Ratus Rupiah Koin", badge: "200", valClass: "val-coin-200", hero: "Dr. Tjipto Mangoenkoesoemo", color: "#cbd5e1" },
    { index: 3, code: "500", nominal: 500, name: "Lima Ratus Rupiah Koin", badge: "500", valClass: "val-coin-500", hero: "Letjen TNI T.B. Simatupang", color: "#e2e8f0" },
    { index: 4, code: "sisi_garuda", nominal: 0, name: "Sisi Garuda / Pahlawan", badge: "Garuda", valClass: "val-coin-garuda", hero: "Lambang Garuda / Pahlawan", color: "#a855f7" }
  ];
  window.COIN_CLASSES = COIN_CLASSES;


  // Ambang Batas Keyakinan Terkalibrasi (Sangat Responsif & Bebas Lockup)
  const THRESHOLDS = {
    CONFIRMED: 0.50,             // Minimal 50% rata-rata confidence untuk konfirmasi
    MARGIN: 0.10,                // Minimal 10% margin di atas peringkat kedua
    UNCERTAIN: 0.35,             // Ambang batas keraguan (ambigu)
    UNKNOWN: 0.30,               // Di bawah 30% dianggap objek tak dikenal / frame kosong
    WINDOW_SIZE: 4,              // Buffer sliding window 4 frame terakhir (responsif ~500ms)
    MIN_VOTES: 3,                // 3 dari 4 frame sepakat untuk konfirmasi cepat
    TARGET_FRAME_INTERVAL_MS: 150, // Target ~6-7 FPS untuk scanning kontinu
    GUIDANCE_COOLDOWN_MS: 4000,  // Jeda minimal antar petunjuk suara (mencegah spam suara)
    SPEECH_SAME_BILL_COOLDOWN_MS: 6000 // Jeda pengucapan ulang bila uang yang sama masih di depan kamera
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

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // -------------------------------------------------------------------------
  // Referensi Elemen DOM
  // -------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const el = {
    // Mode Switcher (Kertas vs Koin)
    modeBanknoteBtn: $("mode-banknote-btn"),
    modeCoinBtn: $("mode-coin-btn"),

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
    camRecoveryUi: $("cam-recovery-ui"),
    btnFixCam: $("btn-fix-cam"),
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

    // Visual Debug Elements (?debug=1)
    debugOverlayCanvas: $("debug-overlay-canvas"),
    debugHud: $("debug-hud"),
    debugFps: $("debug-fps"),
    debugTensorPreview: $("debug-tensor-preview"),
    debugStatus: $("debug-status"),
    debugSource: $("debug-source"),
    debugBbox: $("debug-bbox"),
    debugQuality: $("debug-quality"),
    debugWindow: $("debug-window"),
    debugWinner: $("debug-winner"),
    debugConf: $("debug-conf"),
    debugMargin: $("debug-margin")
  };

  // -------------------------------------------------------------------------
  // Status Aplikasi (State Management)
  // -------------------------------------------------------------------------
  const isDebugUrl = new URLSearchParams(window.location.search).get("debug") === "1";

  const app = {
    // Dual Mode (Banknote vs Coin)
    currentMode: "banknote",
    tfCoinModel: null,
    coinModelReady: false,

    stream: null,
    availableCameras: [],
    activeCameraIndex: 0,
    blackFrameHealAttempts: 0,
    currentFacingMode: "environment", // default environment, fallback user
    torchOn: false,
    zoomLevel: 1,
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
    currentState: "IDLE",

    // Continuous Autonomous Scanning & Consensus State
    continuousScanning: true,
    scanLoopActive: false,
    slidingWindow: [], // Array objek: { nominal, code, prob, margin, timestamp }
    lastSpokenNominal: null,
    lastSpokenTimestamp: 0,
    lastGuidanceTimestamp: 0,
    unconfirmedFramesCount: 0,
    consecutiveEmptyFrames: 0,

    // Debug Mode & Performance Monitoring
    isDebug: isDebugUrl,
    fpsHistory: [],
    lastFrameTimestamp: performance.now(),
    currentFps: 0
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

  function speakGuidance(teks) {
    const now = Date.now();
    if (now - app.lastGuidanceTimestamp < THRESHOLDS.GUIDANCE_COOLDOWN_MS) {
      return;
    }
    if (speechSynthesis && speechSynthesis.speaking) {
      return;
    }
    app.lastGuidanceTimestamp = now;
    speak(teks);
  }

  function announce(pesan) {
    if (!el.srStatus) return;
    el.srStatus.textContent = "";
    setTimeout(() => { el.srStatus.textContent = pesan; }, 40);
  }

  function getar(pola) {
    try {
      if (navigator.userActivation && !navigator.userActivation.hasBeenActive) return;
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
          loaded = await tf.loadLayersModel(loc, {
            requestInit: { cache: "no-store" }
          });
          if (loaded) break;
        } catch (err) {
          console.warn(`[MLOps] Gagal memuat dari ${loc}:`, err);
        }
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
      if (el.diagModel) el.diagModel.textContent = "MobileNetV2 v11.0 Production";

      // Jalankan loop pemindaian kontinu begitu model dan kamera siap
      startContinuousScanLoop();
    } catch (err) {
      console.error("[MLOps Error] Gagal inisialisasi model TF.js:", err);
      if (el.chipText) el.chipText.textContent = "AI Fallback Active";
      if (el.sheetGpuVal) el.sheetGpuVal.textContent = "Fallback Engine";
    }
  }

  // -------------------------------------------------------------------------
  // Perhitungan Koordinat Crop Presisi dari Kotak Viewfinder UI
  // -------------------------------------------------------------------------
  function getViewfinderCropCoords(videoEl, viewfinderEl) {
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (!vw || !vh) {
      return { sx: 0, sy: 0, sw: 224, sh: 224 };
    }

    const vRect = videoEl.getBoundingClientRect ? videoEl.getBoundingClientRect() : { width: vw, height: vh, left: 0, top: 0 };
    const vf = viewfinderEl || el.viewfinder;
    const vfRect = (vf && vf.getBoundingClientRect) ? vf.getBoundingClientRect() : {
      left: vRect.left + vRect.width * 0.15,
      top: vRect.top + vRect.height * 0.20,
      width: vRect.width * 0.70,
      height: vRect.height * 0.55,
      right: vRect.left + vRect.width * 0.85
    };

    const scale = Math.max(vRect.width / vw, vRect.height / vh);
    const renderedW = vw * scale;
    const renderedH = vh * scale;
    const renderedLeft = vRect.left + (vRect.width - renderedW) / 2;
    const renderedTop = vRect.top + (vRect.height - renderedH) / 2;

    const isMirrored = document.body.classList.contains("cam-user") ||
      (app && app.currentFacingMode === "user");

    const screenLeft = isMirrored
      ? renderedW - (vfRect.right - renderedLeft)
      : (vfRect.left - renderedLeft);
    const screenTop = vfRect.top - renderedTop;

    const sxRaw = screenLeft / scale;
    const syRaw = screenTop / scale;
    const swRaw = vfRect.width / scale;
    const shRaw = vfRect.height / scale;

    const sx = Math.max(0, Math.min(vw - 1, Math.round(sxRaw)));
    const sy = Math.max(0, Math.min(vh - 1, Math.round(syRaw)));
    const sw = Math.max(1, Math.min(vw - sx, Math.round(swRaw)));
    const sh = Math.max(1, Math.min(vh - sy, Math.round(shRaw)));

    return { sx, sy, sw, sh };
  }

  // -------------------------------------------------------------------------
  // Lokalisasi Uang Kertas Otomatis (Lightweight Client-Side Saliency)
  // -------------------------------------------------------------------------
  // Menganalisis area foreground secara cerdas tanpa terjebak warna baju di bawah
  let analysisCanvas = null;
  let analysisCtx = null;

  function locateBanknoteCandidate(videoEl) {
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (!vw || !vh) {
      return {
        status: "NO_FRAME",
        qualityStatus: "OK",
        sx: 0, sy: 0, sw: 224, sh: 224,
        confidence: 0,
        aspectRatio: 1.0,
        avgBrightness: 128,
        avgBlurEnergy: 10
      };
    }

    const analysisW = 320;
    const analysisH = Math.max(1, Math.round(analysisW * (vh / vw)));

    if (!analysisCanvas) {
      analysisCanvas = document.createElement("canvas");
      analysisCanvas.width = analysisW;
      analysisCanvas.height = analysisH;
      analysisCtx = analysisCanvas.getContext("2d", { willReadFrequently: true });
    } else if (analysisCanvas.height !== analysisH) {
      analysisCanvas.height = analysisH;
    }

    // Salin frame video penuh ke kanvas analisis beresolusi rendah
    analysisCtx.drawImage(videoEl, 0, 0, analysisW, analysisH);
    const imgData = analysisCtx.getImageData(0, 0, analysisW, analysisH);
    const pixels = imgData.data;

    // Batasi pencarian aktif pada area atas/tengah (maks 76% tinggi frame)
    // agar baju/kerah/kancing di bagian dada bawah tidak mendominasi bounding box
    const searchH = Math.round(analysisH * 0.76);

    let totalLuma = 0;
    let totalGradient = 0;
    let sampleCount = 0;

    const blockSize = 16;
    const gridCols = Math.floor(analysisW / blockSize);
    const gridRows = Math.floor(searchH / blockSize);
    const gridEnergy = new Float32Array(gridCols * gridRows);

    const step = 2; // Step 2 piksel untuk kecepatan (< 4ms di CPU)
    for (let y = 1; y < searchH - 1; y += step) {
      const rowOffset = y * analysisW;
      const gridY = Math.min(gridRows - 1, Math.floor(y / blockSize));

      for (let x = 1; x < analysisW - 1; x += step) {
        const idx = (rowOffset + x) * 4;
        const r = pixels[idx];
        const g = pixels[idx + 1];
        const b = pixels[idx + 2];

        // Luminance Y
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;
        totalLuma += luma;
        sampleCount++;

        // Gradient Magnitude (Perbedaan sentral horizontal & vertikal)
        const idxRight = (rowOffset + x + 1) * 4;
        const idxLeft  = (rowOffset + x - 1) * 4;
        const idxDown  = ((y + 1) * analysisW + x) * 4;
        const idxUp    = ((y - 1) * analysisW + x) * 4;

        const lumaR = 0.299 * pixels[idxRight] + 0.587 * pixels[idxRight + 1] + 0.114 * pixels[idxRight + 2];
        const lumaL = 0.299 * pixels[idxLeft]  + 0.587 * pixels[idxLeft + 1]  + 0.114 * pixels[idxLeft + 2];
        const lumaD = 0.299 * pixels[idxDown]  + 0.587 * pixels[idxDown + 1]  + 0.114 * pixels[idxDown + 2];
        const lumaU = 0.299 * pixels[idxUp]    + 0.587 * pixels[idxUp + 1]    + 0.114 * pixels[idxUp + 2];

        const grad = Math.abs(lumaR - lumaL) + Math.abs(lumaD - lumaU);
        totalGradient += grad;

        // Deteksi Warna Kulit Manusia (Tangan / Jari / Wajah)
        const isSkin = (r > 75 && g > 40 && b > 20 && r > g && (r - b) > 15 && (Math.max(r, g, b) - Math.min(r, g, b) > 15));
        const colorSpread = Math.max(r, g, b) - Math.min(r, g, b);

        let saliency = grad;
        if (isSkin) {
          saliency *= 0.12; // Redam area jari/tangan
        } else {
          // Bobot tinggi pada tekstur guilloche & angka nominal uang kertas
          saliency += colorSpread * 0.35;
        }

        const gridX = Math.min(gridCols - 1, Math.floor(x / blockSize));
        gridEnergy[gridY * gridCols + gridX] += saliency;
      }
    }

    const avgBrightness = totalLuma / Math.max(1, sampleCount);
    const avgBlurEnergy = totalGradient / Math.max(1, sampleCount);

    let qualityStatus = "OK";
    if (avgBrightness < 35) {
      qualityStatus = "DARK";
    } else if (avgBrightness > 225) {
      qualityStatus = "OVEREXPOSED";
    } else if (avgBlurEnergy < 3.5) {
      qualityStatus = "BLURRY";
    }

    // 2. Evaluasi Bounding Box Kandidat Uang
    let totalGrid = 0;
    for (let i = 0; i < gridEnergy.length; i++) totalGrid += gridEnergy[i];
    const meanGrid = totalGrid / Math.max(1, gridEnergy.length);
    const energyThresh = meanGrid * 1.30;

    let minCol = gridCols, maxCol = -1, minRow = gridRows, maxRow = -1;
    let activeBlockCount = 0;

    for (let gy = 0; gy < gridRows; gy++) {
      for (let gx = 0; gx < gridCols; gx++) {
        if (gridEnergy[gy * gridCols + gx] > energyThresh) {
          activeBlockCount++;
          if (gx < minCol) minCol = gx;
          if (gx > maxCol) maxCol = gx;
          if (gy < minRow) minRow = gy;
          if (gy > maxRow) maxRow = gy;
        }
      }
    }

    // Jika blok aktif terlalu sedikit atau mencakup seluruh layar, gunakan default foreground tengah
    let candX, candY, candW, candH;
    const totalBlocks = gridCols * gridRows;
    const coverageRatio = activeBlockCount / Math.max(1, totalBlocks);

    if (activeBlockCount < 4 || minCol > maxCol || minRow > maxRow || coverageRatio > 0.72) {
      // Default: 70% lebar tengah, 55% tinggi atas (posisi alami memegang uang di depan kamera)
      candX = Math.round(analysisW * 0.15);
      candY = Math.round(analysisH * 0.12);
      candW = Math.round(analysisW * 0.70);
      candH = Math.round(analysisH * 0.55);
    } else {
      candX = minCol * blockSize;
      candY = minRow * blockSize;
      candW = (maxCol - minCol + 1) * blockSize;
      candH = (maxRow - minRow + 1) * blockSize;

      // Beri safety margin 10%
      const padX = Math.round(candW * 0.10);
      const padY = Math.round(candH * 0.10);
      candX = Math.max(0, candX - padX);
      candY = Math.max(0, candY - padY);
      candW = Math.min(analysisW - candX, candW + padX * 2);
      candH = Math.min(analysisH - candY, candH + padY * 2);
    }

    const candAreaRatio = (candW * candH) / (analysisW * analysisH);
    const candAspectRatio = candW / Math.max(1, candH);

    let candidateStatus = "OK";
    if (candAreaRatio < 0.05) {
      candidateStatus = "TOO_FAR";
    }

    const scaleX = vw / analysisW;
    const scaleY = vh / analysisH;

    const sx = Math.max(0, Math.min(vw - 1, Math.round(candX * scaleX)));
    const sy = Math.max(0, Math.min(vh - 1, Math.round(candY * scaleY)));
    const sw = Math.max(1, Math.min(vw - sx, Math.round(candW * scaleX)));
    const sh = Math.max(1, Math.min(vh - sy, Math.round(candH * scaleY)));

    return {
      status: candidateStatus,
      qualityStatus,
      avgBrightness,
      avgBlurEnergy,
      coverageRatio: candAreaRatio,
      aspectRatio: candAspectRatio,
      candPixelBox: { x: candX, y: candY, w: candW, h: candH },
      sx, sy, sw, sh,
      confidence: Math.min(1.0, activeBlockCount / 12)
    };
  }

  // -------------------------------------------------------------------------
  // Preprocessing Citra: Mode 'pad' (ImageOps.contain 224x224 RGB [0, 255])
  // -------------------------------------------------------------------------
  // Mendukung horizontal flip (un-mirroring) otomatis untuk kamera depan / webcam
  let preprocessCanvas = null;
  let preprocessCtx = null;

  function extractTensorAndCanvas(videoEl, sx, sy, sw, sh, flipHorizontal = false) {
    if (!preprocessCanvas) {
      preprocessCanvas = document.createElement("canvas");
      preprocessCanvas.width = 224;
      preprocessCanvas.height = 224;
      preprocessCtx = preprocessCanvas.getContext("2d", { willReadFrequently: true });
    }

    // 1. Isi kanvas dengan padding hitam rgb(0,0,0) (persis ImageOps.contain di notebook)
    preprocessCtx.save();
    preprocessCtx.fillStyle = "#000000";
    preprocessCtx.fillRect(0, 0, 224, 224);

    // 2. Mode 'pad': jaga rasio aspek asli uang kertas
    const containScale = Math.min(224 / sw, 224 / sh);
    const destW = Math.max(1, Math.round(sw * containScale));
    const destH = Math.max(1, Math.round(sh * containScale));
    const destX = Math.round((224 - destW) / 2);
    const destY = Math.round((224 - destH) / 2);

    preprocessCtx.imageSmoothingEnabled = true;
    preprocessCtx.imageSmoothingQuality = "high";

    if (flipHorizontal) {
      // Un-mirror front-facing selfie camera agar tulisan nominal & potret pahlawan tidak terbalik
      preprocessCtx.translate(224, 0);
      preprocessCtx.scale(-1, 1);
      preprocessCtx.drawImage(videoEl, sx, sy, sw, sh, 224 - (destX + destW), destY, destW, destH);
    } else {
      preprocessCtx.drawImage(videoEl, sx, sy, sw, sh, destX, destY, destW, destH);
    }
    preprocessCtx.restore();

    // 3. Konversi ke Tensor float32 [1, 224, 224, 3] dalam rentang [0, 255] RGB
    // CATATAN: TIDAK DIBAGI 255.0 (Model sudah memiliki layer internal Rescaling(1./255))
    const tensor = tf.tidy(() => {
      return tf.browser.fromPixels(preprocessCanvas)
        .toFloat()
        .expandDims(0);
    });

    return { tensor, canvas: preprocessCanvas };
  }

  // -------------------------------------------------------------------------
  // Inferensi Multi-Crop v10.1 (Full-Frame + Grid 2×2 — TANPA viewfinder)
  // -------------------------------------------------------------------------
  // Strategi: jalankan beberapa crop berbeda, pilih yang punya score tertinggi.
  // Semua crop diproses dengan mode 'pad' (letterbox hitam) persis seperti training.
  // TIDAK ADA crop dari viewfinder — seluruh frame kamera adalah area deteksi.
  async function predictBanknoteFrame(videoEl) {
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (!vw || !vh || !app.tfModel) return null;

    const t0 = performance.now();
    const isFrontCam = document.body.classList.contains("cam-user") || (app.currentFacingMode === "user");

    // 1. Lokalisasi ringan untuk info kualitas frame & bbox overlay di debug mode
    const loc = locateBanknoteCandidate(videoEl);

    // 2. Susun daftar crop: full-frame + grid 2×2 (overlapping 20%)
    //    Setiap crop ditentukan sebagai (sx, sy, sw, sh) dalam koordinat piksel video asli.
    const cropList = [];

    // Crop 0: Full-Frame — persis input yang diterima model saat training (foto uang landscape utuh)
    cropList.push({ id: "full", sx: 0, sy: 0, sw: vw, sh: vh });

    // Crop 1–4: Grid 2×2 overlapping 20% (60% lebar & tinggi per tile)
    //   Ini memastikan uang yang ada di sudut manapun tetap terwakili utuh dalam setidaknya 1 crop.
    const tileW = Math.round(vw * 0.60);
    const tileH = Math.round(vh * 0.60);
    const stepX = Math.round(vw * 0.40); // 40% step → 20% overlap
    const stepY = Math.round(vh * 0.40);
    for (let gy = 0; gy < 2; gy++) {
      for (let gx = 0; gx < 2; gx++) {
        const sx = Math.min(gx * stepX, vw - tileW);
        const sy = Math.min(gy * stepY, vh - tileH);
        cropList.push({ id: `g${gy}${gx}`, sx, sy, sw: tileW, sh: tileH });
      }
    }

    // Crop 5: Area lokalisasi (hanya jika coverage wajar: tidak terlalu luas dan tidak terlalu kecil)
    if (loc.coverageRatio >= 0.04 && loc.coverageRatio <= 0.65 && loc.status === "OK") {
      cropList.push({ id: "loc", sx: loc.sx, sy: loc.sy, sw: loc.sw, sh: loc.sh });
    }

    // Crop 6: Horizontal Center Crop (khusus saat layar dalam orientasi portrait / HP)
    // Uang kertas yang dipegang melintang di depan layar portrait mendapat resolusi maksimal
    if (vh > vw) {
      const chH = Math.round(vh * 0.48);
      const chW = Math.round(vw * 0.94);
      const chX = Math.round((vw - chW) / 2);
      const chY = Math.round((vh - chH) / 2);
      cropList.push({ id: "center_h", sx: chX, sy: chY, sw: chW, sh: chH });
    }

    // 3. Jalankan semua crop melalui model, kumpulkan skor
    let bestResult = null;
    let bestScore = -1;

    // Gunakan satu canvas yang sudah ada (extractTensorAndCanvas memakai preprocessCanvas singleton)
    // Proses satu per satu untuk menghindari race condition pada canvas bersama.
    for (const crop of cropList) {
      const { tensor, canvas } = extractTensorAndCanvas(
        videoEl, crop.sx, crop.sy, crop.sw, crop.sh, isFrontCam
      );

      let probs;
      try {
        const pred = app.tfModel.predict(tensor);
        probs = Array.from(await pred.data());
        pred.dispose();
      } finally {
        tensor.dispose();
      }

      const ranked = probs
        .map((prob, idx) => ({ ...BANKNOTE_CLASSES[idx], prob }))
        .sort((a, b) => b.prob - a.prob);

      const top = ranked[0];
      const second = ranked[1];
      const margin = top.prob - second.prob;

      // Score = prob × (1 + margin): menggabungkan confidence dan ketegasan prediksi
      const score = top.prob * (1 + margin);

      if (score > bestScore) {
        bestScore = score;
        bestResult = {
          source: crop.id,
          top,
          second,
          margin,
          ranked,
          rawProbs: probs,
          canvas   // canvas dari extractTensorAndCanvas adalah preprocessCanvas singleton
        };
      }
    }

    const latency = Math.round(performance.now() - t0);
    app.lastLatencyMs = latency;

    return {
      loc,
      chosen: bestResult,
      latency
    };
  }

  // -------------------------------------------------------------------------
  // Mode Debug Visual (?debug=1 di URL)
  // -------------------------------------------------------------------------
  function renderDebugOverlay(loc, chosen, latency, fps) {
    if (!app.isDebug || !el.debugOverlayCanvas || !el.debugHud) return;

    el.debugOverlayCanvas.hidden = false;
    el.debugHud.hidden = false;

    const overlay = el.debugOverlayCanvas;
    const v = el.video;
    const vRect = v.getBoundingClientRect();

    if (overlay.width !== vRect.width || overlay.height !== vRect.height) {
      overlay.width = vRect.width;
      overlay.height = vRect.height;
    }

    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (loc && loc.sx !== undefined && v.videoWidth) {
      // Petakan koordinat video asli ke koordinat layar kanvas
      const scale = Math.max(vRect.width / v.videoWidth, vRect.height / v.videoHeight);
      const renderedW = v.videoWidth * scale;
      const renderedH = v.videoHeight * scale;
      const leftOffset = (vRect.width - renderedW) / 2;
      const topOffset = (vRect.height - renderedH) / 2;

      const isMirrored = document.body.classList.contains("cam-user") || (app.currentFacingMode === "user");

      let boxX = leftOffset + loc.sx * scale;
      if (isMirrored) {
        boxX = vRect.width - (leftOffset + (loc.sx + loc.sw) * scale);
      }
      const boxY = topOffset + loc.sy * scale;
      const boxW = loc.sw * scale;
      const boxH = loc.sh * scale;

      // Gambar Bounding Box Hijau Neon di Sekitar Kandidat Uang
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = loc.status === "OK" ? "#22c55e" : "#f59e0b";
      ctx.fillStyle = loc.status === "OK" ? "rgba(34, 197, 94, 0.12)" : "rgba(245, 158, 11, 0.12)";
      ctx.strokeRect(boxX, boxY, boxW, boxH);
      ctx.fillRect(boxX, boxY, boxW, boxH);

      // Label Bounding Box
      ctx.font = "bold 11px monospace";
      ctx.fillStyle = "#ffffff";
      const tagText = `${chosen ? chosen.top.badge : "Target"} (${loc.aspectRatio.toFixed(2)}:1)`;
      const textW = ctx.measureText(tagText).width;
      ctx.fillStyle = loc.status === "OK" ? "#15803d" : "#b45309";
      ctx.fillRect(boxX, Math.max(0, boxY - 18), textW + 10, 18);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(tagText, boxX + 5, Math.max(13, boxY - 4));
    }

    // Perbarui PIP Thumbnail Canvas 224x224
    if (chosen && chosen.canvas && el.debugTensorPreview) {
      const pCtx = el.debugTensorPreview.getContext("2d");
      pCtx.drawImage(chosen.canvas, 0, 0, 224, 224);
    }

    // Perbarui HUD Metrics
    if (el.debugFps) el.debugFps.textContent = `${fps} FPS`;
    if (el.debugStatus) el.debugStatus.textContent = loc ? loc.status : "-";
    if (el.debugSource) el.debugSource.textContent = chosen ? chosen.source : "-";
    if (el.debugBbox && loc) el.debugBbox.textContent = `${loc.sw}×${loc.sh}`;
    if (el.debugQuality && loc) el.debugQuality.textContent = `${loc.qualityStatus} (L:${Math.round(loc.avgBrightness)})`;

    // Riwayat 6 Frame
    if (el.debugWindow) {
      const winStr = app.slidingWindow.map(f => {
        const k = f.nominal >= 1000 ? `${f.nominal / 1000}k` : f.nominal;
        return k;
      }).join(",");
      el.debugWindow.textContent = winStr || "[]";
    }

    if (chosen) {
      if (el.debugWinner) el.debugWinner.textContent = `Rp ${chosen.top.badge}`;
      if (el.debugConf) el.debugConf.textContent = `${(chosen.top.prob * 100).toFixed(1)}%`;
      if (el.debugMargin) el.debugMargin.textContent = `+${(chosen.margin * 100).toFixed(1)}%`;
    }
  }

  // -------------------------------------------------------------------------
  // Render MLOps Diagnostic Panel (Di dalam Bottom Sheet Menu)
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
  // UI Helpers (Animasi Sukses & Pembaruan Kartu)
  // -------------------------------------------------------------------------
  function showSuccessBloom() {
    document.body.classList.remove("state-scanning");
    document.body.classList.add("show-success");
    setTimeout(() => {
      document.body.classList.remove("show-success");
    }, 750);
  }

  function updateResultUI(banknote) {
    if (el.banknoteThumb && el.thumbNumber) {
      el.banknoteThumb.className = `banknote-thumbnail ${banknote.valClass}`;
      el.thumbNumber.textContent = banknote.badge;
    }

    el.bottomCard.className = "bottom-pill-card state-result";
    el.pillSubtitle.textContent = "Hasil Deteksi";
    el.pillTitle.textContent = formatRupiah(banknote.nominal);

    if (app.resetCardTimer) clearTimeout(app.resetCardTimer);
    app.resetCardTimer = setTimeout(() => {
      // Jika tidak ada uang lagi di frame, kembali ke state idle
      if (!app.lastSpokenNominal) {
        el.bottomCard.className = "bottom-pill-card state-idle";
        el.pillSubtitle.textContent = "Scanning...";
        el.pillTitle.textContent = "Arahkan Uang ke Kamera";
      }
    }, 4500);
  }

  // -------------------------------------------------------------------------
  // Loop Pemindaian Kontinu & Konsensus Sliding Window (Aksesibilitas Tunanetra)
  // -------------------------------------------------------------------------
  async   // -------------------------------------------------------------------------
  // Inisialisasi Model Koin In-Browser TensorFlow.js (MobileNetV2 5-Class)
  // -------------------------------------------------------------------------
  async function initCoinModel() {
    if (typeof tf === "undefined") {
      setTimeout(initCoinModel, 400);
      return;
    }

    try {
      await tf.ready();
      const coinModelLocations = [
        "./model_coin_tfjs/model.json",
        "model_coin_tfjs/model.json",
        "../model_coin_tfjs/model.json"
      ];

      let loaded = null;
      for (const loc of coinModelLocations) {
        try {
          console.log(`[MLOps] Memuat model koin dari: ${loc}`);
          loaded = await tf.loadLayersModel(loc, {
            requestInit: { cache: "no-store" }
          });
          if (loaded) break;
        } catch (err) {
          console.warn(`[MLOps] Percobaan memuat koin dari ${loc} gagal:`, err);
        }
      }

      if (!loaded) {
        throw new Error("Tidak dapat menemukan model_coin_tfjs/model.json.");
      }

      app.tfCoinModel = loaded;
      app.coinModelReady = true;

      // Warmup forward pass untuk kompilasi shader WebGL
      tf.tidy(() => {
        const dummy = tf.zeros([1, 224, 224, 3]);
        app.tfCoinModel.predict(dummy);
      });

      console.log("[MLOps] MobileNetV2 5-Class (Koin) siap inferensi!");
    } catch (err) {
      console.warn("[MLOps] Error inisialisasi model koin:", err);
    }
  }

  // -------------------------------------------------------------------------
  // Inferensi Frame Khusus Uang Logam (Koin)
  // -------------------------------------------------------------------------
  async function predictCoinFrame(videoEl) {
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (!vw || !vh || !app.tfCoinModel) return null;

    const t0 = performance.now();
    const isFrontCam = document.body.classList.contains("cam-user") || (app.currentFacingMode === "user");

    // Crop terpusat 1:1 pada area viewfinder koin
    const minDim = Math.min(vw, vh);
    const cropSize = Math.round(minDim * 0.65);
    const sx = Math.max(0, Math.round((vw - cropSize) / 2));
    const sy = Math.max(0, Math.round((vh - cropSize) / 2));

    const tensorData = extractTensorAndCanvas(videoEl, sx, sy, cropSize, cropSize, isFrontCam);
    if (!tensorData) return null;

    const { tensor } = tensorData;
    let probs = null;
    try {
      const predTensor = app.tfCoinModel.predict(tensor);
      probs = await predTensor.data();
      predTensor.dispose();
    } finally {
      tensor.dispose();
    }

    const latency = Math.round(performance.now() - t0);
    app.lastLatencyMs = latency;

    const ranked = Array.from(probs)
      .map((prob, idx) => ({ ...COIN_CLASSES[idx], prob }))
      .sort((a, b) => b.prob - a.prob);

    const top = ranked[0];
    const second = ranked[1] || { prob: 0 };
    const margin = top.prob - second.prob;

    return {
      top,
      second,
      margin,
      ranked,
      latency,
      cropBox: { sx, sy, sw: cropSize, sh: cropSize }
    };
  }

  // -------------------------------------------------------------------------
  // Pergantian Mode Deteksi (Uang Kertas vs Uang Logam)
  // -------------------------------------------------------------------------
  function switchDetectionMode(mode) {
    if (app.currentMode === mode) return;
    app.currentMode = mode;
    app.slidingWindow = [];
    app.lastSpokenNominal = null;
    app.lastConfirmedNominal = null;
    app.consecutiveEmptyFrames = 0;

    if (mode === "coin") {
      document.body.classList.add("mode-coin");
      if (el.modeBanknoteBtn) {
        el.modeBanknoteBtn.classList.remove("active");
        el.modeBanknoteBtn.setAttribute("aria-selected", "false");
      }
      if (el.modeCoinBtn) {
        el.modeCoinBtn.classList.add("active");
        el.modeCoinBtn.setAttribute("aria-selected", "true");
      }

      if (el.bottomCard) {
        el.bottomCard.className = "bottom-pill-card state-idle";
        el.pillSubtitle.textContent = "Mode Uang Logam";
        el.pillTitle.textContent = "Arahkan Koin ke Kamera";
      }
      if (el.banknoteThumb && el.thumbNumber) {
        el.banknoteThumb.className = "banknote-thumbnail val-scanning";
        el.thumbNumber.textContent = "🪙";
      }

      getar([80]);
      speak("Mode deteksi uang logam aktif, arahkan koin ke kamera.");

      if (!app.coinModelReady) {
        initCoinModel();
      }
    } else {
      document.body.classList.remove("mode-coin");
      if (el.modeBanknoteBtn) {
        el.modeBanknoteBtn.classList.add("active");
        el.modeBanknoteBtn.setAttribute("aria-selected", "true");
      }
      if (el.modeCoinBtn) {
        el.modeCoinBtn.classList.remove("active");
        el.modeCoinBtn.setAttribute("aria-selected", "false");
      }

      if (el.bottomCard) {
        el.bottomCard.className = "bottom-pill-card state-idle";
        el.pillSubtitle.textContent = "Mode Uang Kertas";
        el.pillTitle.textContent = "Arahkan Uang ke Kamera";
      }
      if (el.banknoteThumb && el.thumbNumber) {
        el.banknoteThumb.className = "banknote-thumbnail val-scanning";
        el.thumbNumber.textContent = "💵";
      }

      getar([80]);
      speak("Mode deteksi uang kertas aktif.");
    }
  }

  async function startContinuousScanLoop() {
    if (app.scanLoopActive) return;
    app.scanLoopActive = true;

    while (app.continuousScanning) {
      const loopStart = performance.now();

      // Hitung FPS real-time
      app.currentFps = Math.round(1000 / Math.max(1, loopStart - app.lastFrameTimestamp));
      app.lastFrameTimestamp = loopStart;

      // Tunggu hingga video kamera dan setidaknya satu model siap
      if (!el.video || !el.video.videoWidth || (!app.modelReady && !app.coinModelReady)) {
        await sleep(200);
        continue;
      }

      // =========================================================================
      // INFERENSI ENSEMBLE PARALEL OTOMATIS (ZERO-TOUCH MULTI-MODAL RECOGNITION)
      // Pengguna tunanetra TIDAK PERLU menekan tombol apa pun. Model secara mandiri
      // mengevaluasi apakah objek adalah uang kertas atau uang logam dalam satu loop.
      // =========================================================================
      try {
        const [banknoteData, coinData] = await Promise.all([
          app.modelReady ? predictBanknoteFrame(el.video) : Promise.resolve(null),
          app.coinModelReady ? predictCoinFrame(el.video) : Promise.resolve(null)
        ]);

        const now = Date.now();
        // Kadaluwarsa vote lama > 700ms (cegah voting lockup)
        app.slidingWindow = app.slidingWindow.filter(item => (now - item.timestamp) < 700);

        // Ekstraksi kandidat teratas dari masing-masing model
        const bTop = banknoteData?.chosen?.top;
        const bMargin = banknoteData?.chosen?.margin || 0;
        const bScore = bTop ? (bTop.prob + 0.45 * bMargin) : 0;

        const cTop = coinData?.top;
        const cMargin = coinData?.margin || 0;
        const cScore = cTop ? (cTop.prob + 0.45 * cMargin) : 0;

        let frameCandidate = null;

        // Aturan Arbitrasi AI Cerdas:
        // 1. Sisi Garuda adalah ciri khas mutlak koin logam
        if (cTop && cTop.code === "sisi_garuda" && cTop.prob >= 0.42) {
          frameCandidate = { type: "coin", ...cTop, margin: cMargin, prob: cTop.prob };
        }
        // 2. Jika skor model koin signifikan lebih unggul
        else if (cScore >= 0.45 && cScore > bScore) {
          frameCandidate = { type: "coin", ...cTop, margin: cMargin, prob: cTop.prob };
        }
        // 3. Jika skor model uang kertas signifikan lebih unggul
        else if (bScore >= 0.38 && bScore > cScore) {
          frameCandidate = { type: "banknote", ...bTop, margin: bMargin, prob: bTop.prob };
        }

        // Tampilkan debug overlay jika aktif
        if (banknoteData && banknoteData.loc && banknoteData.chosen) {
          renderDebugOverlay(banknoteData.loc, banknoteData.chosen, banknoteData.latency, app.currentFps);
        }

        if (frameCandidate) {
          // Deteksi pergantian uang secara instan
          if (app.lastConfirmedNominal && frameCandidate.nominal !== app.lastConfirmedNominal) {
            app.slidingWindow = app.slidingWindow.filter(item => item.nominal === frameCandidate.nominal && item.type === frameCandidate.type);
            app.lastSpokenNominal = null;
          }

          app.slidingWindow.push({
            type: frameCandidate.type,
            nominal: frameCandidate.nominal,
            code: frameCandidate.code,
            name: frameCandidate.name,
            badge: frameCandidate.badge,
            color: frameCandidate.color,
            hero: frameCandidate.hero,
            prob: frameCandidate.prob,
            timestamp: now
          });

          if (app.slidingWindow.length > THRESHOLDS.WINDOW_SIZE) {
            app.slidingWindow.shift();
          }
          app.consecutiveEmptyFrames = 0;

          // Adaptasi bentuk viewfinder secara dinamis
          if (frameCandidate.type === "coin") {
            document.body.classList.add("mode-coin");
          } else {
            document.body.classList.remove("mode-coin");
          }
        } else {
          app.consecutiveEmptyFrames++;
          if (app.consecutiveEmptyFrames >= 3) {
            app.lastSpokenNominal = null;
            app.lastConfirmedNominal = null;
            app.slidingWindow = [];
            document.body.classList.remove("mode-coin");

            if (el.bottomCard && el.bottomCard.classList.contains("state-result")) {
              el.bottomCard.className = "bottom-pill-card state-idle";
              el.pillSubtitle.textContent = "Scanning Otomatis...";
              el.pillTitle.textContent = "Arahkan Uang ke Kamera";
              if (el.banknoteThumb && el.thumbNumber) {
                el.banknoteThumb.className = "banknote-thumbnail val-scanning";
                el.thumbNumber.textContent = "...";
              }
            }
          }
        }

        // =========================================================================
        // KONSENSUS TEMPORAL MULTI-MODAL (MINIMAL 2 FRAME KONSISTEN)
        // =========================================================================
        if (app.slidingWindow.length >= 2) {
          const voteCounts = {};
          const sumProb = {};
          for (const vote of app.slidingWindow) {
            const key = `${vote.type}_${vote.code}`;
            voteCounts[key] = (voteCounts[key] || 0) + 1;
            sumProb[key] = (sumProb[key] || 0) + vote.prob;
          }

          let winKey = null;
          let maxVotes = 0;
          for (const [key, count] of Object.entries(voteCounts)) {
            if (count > maxVotes) {
              maxVotes = count;
              winKey = key;
            }
          }

          const avgProb = sumProb[winKey] / maxVotes;
          if (maxVotes >= 2 && avgProb >= 0.40) {
            const [winType, winCode] = winKey.split("_");
            const winVote = app.slidingWindow.find(v => v.type === winType && v.code === winCode);

            if (winType === "coin") {
              // -------------------------------------------------------------
              // HASIL DETEKSI UANG KOIN
              // -------------------------------------------------------------
              if (winCode === "sisi_garuda") {
                if (el.bottomCard) {
                  el.bottomCard.className = "bottom-pill-card state-result";
                  el.pillSubtitle.textContent = "🪙 Koin: Sisi Pahlawan / Garuda";
                  el.pillTitle.textContent = "Balik Koin untuk Membaca Nominal";
                  if (el.banknoteThumb && el.thumbNumber) {
                    el.banknoteThumb.className = "banknote-thumbnail val-coin-garuda";
                    el.thumbNumber.textContent = "🦅";
                  }
                }
                if (now - app.lastSpokenTimestamp > 4500) {
                  app.lastSpokenTimestamp = now;
                  getar([80, 50, 80]);
                  speak("Sisi gambar Garuda atau Pahlawan, silakan balik koin untuk memastikan angka nominal.");
                }
              } else {
                const nominal = winVote.nominal;
                app.lastConfirmedNominal = nominal;
                app.lastResult = { nominal, isCoin: true };

                if (el.bottomCard) {
                  el.bottomCard.className = "bottom-pill-card state-result";
                  el.pillSubtitle.textContent = `🪙 Uang Logam Terdeteksi (${Math.round(avgProb * 100)}%)`;
                  el.pillTitle.textContent = `Rp ${nominal.toLocaleString("id-ID")}`;
                  if (el.banknoteThumb && el.thumbNumber) {
                    el.banknoteThumb.className = `banknote-thumbnail val-coin-${nominal}`;
                    el.thumbNumber.textContent = winVote.badge;
                  }
                }

                triggerSuccessBadge();

                if (now - app.lastSpokenTimestamp > THRESHOLDS.SPEECH_SAME_BILL_COOLDOWN_MS || app.lastSpokenNominal !== nominal) {
                  app.lastSpokenNominal = nominal;
                  app.lastSpokenTimestamp = now;
                  getar([120]);
                  speak(nominalToSpeech(nominal, true));
                }
              }
            } else {
              // -------------------------------------------------------------
              // HASIL DETEKSI UANG KERTAS
              // -------------------------------------------------------------
              const nominal = winVote.nominal;
              app.lastConfirmedNominal = nominal;
              app.lastResult = { nominal, isCoin: false };

              if (el.bottomCard) {
                el.bottomCard.className = "bottom-pill-card state-result";
                el.pillSubtitle.textContent = `💵 Uang Kertas Terdeteksi (${Math.round(avgProb * 100)}%)`;
                el.pillTitle.textContent = `Rp ${nominal.toLocaleString("id-ID")}`;
                if (el.banknoteThumb && el.thumbNumber) {
                  el.banknoteThumb.className = `banknote-thumbnail val-${nominal}`;
                  el.thumbNumber.textContent = winVote.badge;
                }
              }

              triggerSuccessBadge();

              if (now - app.lastSpokenTimestamp > THRESHOLDS.SPEECH_SAME_BILL_COOLDOWN_MS || app.lastSpokenNominal !== nominal) {
                app.lastSpokenNominal = nominal;
                app.lastSpokenTimestamp = now;
                getar([120]);
                speak(nominalToSpeech(nominal, false));
              }
            }
          }
        }
      } catch (err) {
        console.warn("[MLOps] Error scan frame:", err);
      }

      const elapsed = performance.now() - loopStart;
      const sleepTime = Math.max(15, THRESHOLDS.TARGET_FRAME_INTERVAL_MS - elapsed);
      await sleep(sleepTime);
    }
  }

  // -------------------------------------------------------------------------
  // Interaksi Manual: Ketuk Layar (Single Tap = Re-Scan & Force Speak)
  // -------------------------------------------------------------------------
  async function forceScanAndSpeak() {
    getar(35);
    app.lastSpokenNominal = null; // Reset kunci pengucapan agar membaca ulang
    app.unconfirmedFramesCount = 0;

    el.bottomCard.className = "bottom-pill-card state-scanning";
    el.pillSubtitle.textContent = "Scanning...";
    el.pillTitle.textContent = "Menganalisis uang...";

    if (!app.tfModel) {
      speak("Memuat model kecerdasan buatan, mohon tunggu sebentar.");
      return;
    }

    // Ambil beberapa frame cepat untuk konfirmasi instan
    let bestPred = null;
    let highestConf = -1;

    for (let i = 0; i < 3; i++) {
      const res = await predictBanknoteFrame(el.video);
      if (res && res.chosen && res.chosen.top.prob > highestConf) {
        highestConf = res.chosen.top.prob;
        bestPred = res.chosen;
      }
      await sleep(60);
    }

    if (bestPred && bestPred.top.prob >= THRESHOLDS.CONFIRMED && bestPred.margin >= THRESHOLDS.MARGIN) {
      const selected = bestPred.top;
      app.lastSpokenNominal = selected.nominal;
      app.lastResult = {
        ...selected,
        type: "banknote",
        confidence: bestPred.top.prob,
        displayText: formatRupiah(selected.nominal),
        speechText: nominalToSpeech(selected.nominal)
      };

      getar([80, 40, 80]);
      showSuccessBloom();
      speak(nominalToSpeech(selected.nominal));
      updateResultUI(selected);
    } else if (bestPred && bestPred.top.prob >= THRESHOLDS.UNCERTAIN) {
      el.bottomCard.className = "bottom-pill-card state-idle";
      el.pillSubtitle.textContent = "Kurang Jelas";
      el.pillTitle.textContent = "Ubah Sudut Uang";
      getar([60, 40, 60]);
      speak("Uang belum terbaca jelas. Mohon ubah sedikit sudut uang.");
    } else {
      el.bottomCard.className = "bottom-pill-card state-idle";
      el.pillSubtitle.textContent = "Tidak Terdeteksi";
      el.pillTitle.textContent = "Dekatkan Uang ke Kamera";
      getar([100, 50, 100]);
      speak("Uang tidak terdeteksi. Dekatkan uang ke kamera lalu coba lagi.");
    }
  }

  // -------------------------------------------------------------------------
  // Akses Kamera Penuh & Switch Kamera
  // -------------------------------------------------------------------------
  // -------------------------------------------------------------------------
  // Inisialisasi & Pengalihan Kamera (Multi-Tier Fallback untuk Laptop & HP)
  // Smart Real-Device Selector & Anti-Virtual/IR Black-Screen Recovery
  // -------------------------------------------------------------------------
  function scoreCameraDevice(device) {
    const label = (device.label || "").toLowerCase();
    if (!label) return 0; // Belum ada izin, tidak dapat dinilai berdasarkan label
    let score = 50;

    // Perangkat Virtual (Cast, OBS, Dummy) atau sensor IR (Infrared/Windows Hello)
    // selalu menghasilkan frame hitam di browser tanpa proyektor IR
    if (/cast|virtual|obs|dummy|software|ir\b|infrared|depth|hello/i.test(label)) {
      score -= 100;
    }
    // Kamera hardware USB / UVC / Laptop RGB
    if (/uvc|usb|webcam|integrated|camera|hd|rgb|facetime/i.test(label)) {
      score += 80;
    }
    // Identitas hardware spesifik (e.g. 2b7e:b685 HD WebCam pada laptop user)
    if (/2b7e|b685|chicony|realtek|sunplus|logitech/i.test(label)) {
      score += 50;
    }
    return score;
  }

  async function getCameraDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      return [];
    }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === "videoinput");

      // Urutkan: kamera hardware fisik berperingkat tertinggi berada di posisi pertama
      videoInputs.sort((a, b) => scoreCameraDevice(b) - scoreCameraDevice(a));
      app.availableCameras = videoInputs;
      console.log("[MLOps Camera] Kamera terdeteksi:", app.availableCameras.map(c => `${c.label || 'Kamera (Izin Tertunda)'} [${c.deviceId ? c.deviceId.slice(0, 8) : 'unknown'}] (Skor: ${scoreCameraDevice(c)})`));
      return app.availableCameras;
    } catch (err) {
      console.warn("[MLOps Camera] Gagal enumerateDevices:", err);
      return [];
    }
  }

  async function startCamera(facingMode = "environment", explicitDeviceId = null, forceNext = false) {
    if (app.stream) {
      try {
        app.stream.getTracks().forEach((t) => t.stop());
      } catch (_) {}
      app.stream = null;
    }

    const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    const targetFacing = isMobile ? facingMode : "user";
    app.currentFacingMode = targetFacing;

    if (document.body) {
      document.body.classList.toggle("cam-user", targetFacing === "user");
    }

    if (el.video) {
      el.video.muted = true;
      el.video.defaultMuted = true;
      el.video.setAttribute("muted", "");
      el.video.setAttribute("playsinline", "");
      el.video.setAttribute("webkit-playsinline", "");
    }

    // Refresh daftar kamera yang tersedia di sistem
    await getCameraDevices();

    let targetDeviceId = explicitDeviceId;
    if (forceNext && app.availableCameras.length > 1) {
      app.activeCameraIndex = (app.activeCameraIndex + 1) % app.availableCameras.length;
      targetDeviceId = app.availableCameras[app.activeCameraIndex]?.deviceId;
    } else if (!targetDeviceId) {
      // Cari kamera yang memiliki label hardware valid dan bukan virtual (jika izin sudah ada)
      const physicalCams = app.availableCameras.filter(c => scoreCameraDevice(c) > 0);
      if (physicalCams.length > 0) {
        if (app.activeCameraIndex >= physicalCams.length) app.activeCameraIndex = 0;
        targetDeviceId = physicalCams[app.activeCameraIndex]?.deviceId;
      }
    }

    let stream = null;
    let usedDevice = null;

    // Tahap 1: Coba dengan targetDeviceId jika ditentukan secara eksplisit
    if (targetDeviceId) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: targetDeviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        usedDevice = app.availableCameras.find((c) => c.deviceId === targetDeviceId);
        console.log("[MLOps Camera] Berhasil membuka kamera via exact deviceId:", usedDevice?.label || targetDeviceId);
      } catch (errExact) {
        console.warn("[MLOps Camera] Gagal exact deviceId, mencoba ideal deviceId:", errExact);
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { ideal: targetDeviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          });
          usedDevice = app.availableCameras.find((c) => c.deviceId === targetDeviceId);
        } catch (_) {}
      }
    }

    // Tahap 2: Jika belum ada targetDeviceId (startup pertama ketika label belum diketahui oleh browser),
    // gunakan facingMode ideal (W3C standard). Browser otomatis memilih Webcam fisik utama.
    if (!stream) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: targetFacing },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        console.log("[MLOps Camera] Berhasil membuka kamera via facingMode ideal:", targetFacing);
      } catch (errFacing) {
        console.warn("[MLOps Camera] Gagal facingMode ideal, fallback generic video: true:", errFacing);
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
        } catch (errGeneric) {
          console.error("[MLOps Camera] Semua percobaan akses kamera belum diizinkan:", errGeneric);
          if (el.chipText) el.chipText.textContent = "Kamera Belum Aktif";
          if (el.pillSubtitle) el.pillSubtitle.textContent = "Izin Kamera Terdeteksi";
          if (el.pillTitle) el.pillTitle.textContent = "Ketuk Layar untuk Membuka Kamera";
          showCameraRecoveryUI();
          return;
        }
      }
    }

    if (stream && el.video) {
      app.stream = stream;
      el.video.srcObject = stream;
      el.video.muted = true;

      // SETELAH STREAM DIBUKA: Izin browser sudah 100% granted, sekarang seluruh label terbaca!
      const activeCams = await getCameraDevices();
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings ? track.getSettings() : {};
      const activeLabel = (track.label || "").toLowerCase();

      // Cek apakah kamera yang aktif secara keliru adalah kamera virtual / IR
      const isVirtualOrBad = /cast|virtual|obs|dummy|software|ir\b|infrared/i.test(activeLabel);
      const physicalAlt = activeCams.find(c => scoreCameraDevice(c) >= 50 && c.deviceId !== settings.deviceId);

      if (isVirtualOrBad && physicalAlt) {
        console.warn(`[MLOps Camera] Terdeteksi kamera virtual ("${track.label}"). Mengalihkan otomatis ke webcam hardware: "${physicalAlt.label}"...`);
        speak("Mengalihkan ke webcam utama laptop...");
        try {
          track.stop();
          const altStream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: physicalAlt.deviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          });
          app.stream = altStream;
          el.video.srcObject = altStream;
          usedDevice = physicalAlt;
        } catch (altErr) {
          console.warn("[MLOps Camera] Gagal mengalihkan otomatis ke hardware:", altErr);
        }
      } else {
        usedDevice = activeCams.find(c => c.deviceId === settings.deviceId) || { label: track.label };
      }

      if (el.sheetCamVal) {
        el.sheetCamVal.textContent = usedDevice?.label || (targetFacing === "user" ? "Kamera Depan / Webcam" : "Kamera Belakang");
      }

      const playStream = async () => {
        try {
          await el.video.play();
          console.log(`[MLOps Camera] Kamera aktif: ${el.video.videoWidth}x${el.video.videoHeight}`);
          if (el.pillTitle && el.pillTitle.textContent.includes("Ketuk Layar")) {
            el.pillSubtitle.textContent = "Scanning Otomatis...";
            el.pillTitle.textContent = "Arahkan Uang ke Kamera";
          }
          hideCameraRecoveryUI();
        } catch (playErr) {
          console.warn("[MLOps Camera] Autoplay tertunda, menunggu gestur:", playErr);
        }
      };

      el.video.onloadedmetadata = playStream;
      try {
        await el.video.play();
      } catch (playErr) {
        console.warn("[MLOps Camera] Autoplay dicegah browser, menunggu gestur pengguna:", playErr);
      }

      if (el.chipText) {
        el.chipText.textContent = app.modelReady ? "AI Scanner Active" : "Camera Ready";
      }

      // Aktifkan pemeriksaan mandiri frame visual (Watchdog Self-Healing)
      scheduleBlackFrameCheck();
    }
  }

  let blackFrameCheckTimer = null;
  function scheduleBlackFrameCheck() {
    if (blackFrameCheckTimer) clearTimeout(blackFrameCheckTimer);
    blackFrameCheckTimer = setTimeout(() => {
      checkAndHealBlackFrame();
    }, 1100);
  }

  async function checkAndHealBlackFrame() {
    if (!el.video || !app.stream) return;

    // Periksa apakah video macet (videoWidth 0) atau paused
    const isFrozen = (el.video.videoWidth === 0 || el.video.videoHeight === 0 || el.video.paused);
    let isPitchBlack = false;

    if (!isFrozen) {
      try {
        const sampleCanvas = document.createElement("canvas");
        sampleCanvas.width = 16;
        sampleCanvas.height = 16;
        const ctx = sampleCanvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(el.video, 0, 0, 16, 16);
        const imgData = ctx.getImageData(0, 0, 16, 16).data;

        let totalLuminance = 0;
        let maxPixel = 0;
        for (let i = 0; i < imgData.length; i += 4) {
          const lum = 0.299 * imgData[i] + 0.587 * imgData[i + 1] + 0.114 * imgData[i + 2];
          totalLuminance += lum;
          if (lum > maxPixel) maxPixel = lum;
        }
        const avgLum = totalLuminance / (16 * 16);

        // Jika rata-rata luminance < 1.0 dan max < 3, kamera ini menghasilkan sinyal kosong / hitam total
        if (avgLum < 1.0 && maxPixel < 3) {
          isPitchBlack = true;
          console.warn(`[MLOps Camera] Terdeteksi stream hitam pekat (Luminance: ${avgLum.toFixed(2)}).`);
        }
      } catch (e) {
        console.warn("[MLOps Camera] Gagal verifikasi frame visual:", e);
      }
    }

    if (isFrozen || isPitchBlack) {
      console.warn(`[MLOps Camera] Masalah visual terdeteksi (Frozen: ${isFrozen}, PitchBlack: ${isPitchBlack}). Percobaan pemulihan #${(app.blackFrameHealAttempts || 0) + 1}...`);
      if ((app.blackFrameHealAttempts || 0) < (app.availableCameras?.length || 2)) {
        app.blackFrameHealAttempts = (app.blackFrameHealAttempts || 0) + 1;
        speak("Mengalihkan ke kamera fisik webcam...");
        await startCamera(app.currentFacingMode, null, true);
        return;
      }
      showCameraRecoveryUI();
    } else {
      app.blackFrameHealAttempts = 0;
      hideCameraRecoveryUI();
      console.log(`[MLOps Camera] Stream visual aktif terverifikasi (${el.video.videoWidth}x${el.video.videoHeight}). Kamera berfungsi prima.`);
    }
  }

  function showCameraRecoveryUI() {
    if (el.camRecoveryUi) {
      el.camRecoveryUi.hidden = false;
    }
  }

  function hideCameraRecoveryUI() {
    if (el.camRecoveryUi) {
      el.camRecoveryUi.hidden = true;
    }
  }

  async function switchCamera() {
    getar(35);
    await getCameraDevices();

    if (app.availableCameras && app.availableCameras.length > 1) {
      // Siklus multi-kamera (Webcam Fisik, Kamera Belakang, dsb)
      app.activeCameraIndex = (app.activeCameraIndex + 1) % app.availableCameras.length;
      const targetCam = app.availableCameras[app.activeCameraIndex];
      const camName = targetCam.label || (`Kamera ${app.activeCameraIndex + 1}`);
      speak(`Beralih ke ${camName}`);
      await startCamera(app.currentFacingMode, targetCam.deviceId);
    } else {
      const newFacing = app.currentFacingMode === "environment" ? "user" : "environment";
      app.currentFacingMode = newFacing;
      speak(newFacing === "user" ? "Beralih ke kamera depan" : "Beralih ke kamera belakang");
      await startCamera(newFacing);
    }
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

    const mirror = app.currentFacingMode === "user" ? " scaleX(-1)" : "";
    const scale = app.zoomLevel === 2 ? " scale(1.3)" : " scale(1)";
    el.video.style.transform = `translate(-50%, -50%)${mirror}${scale}`;
    el.video.style.transition = "transform 0.3s ease";
  }

  // -------------------------------------------------------------------------
  // Akumulasi Total Uang (Hanya via Double Tap / Menu)
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
  // Dispatcher Ketuk Layar (Single Tap = Re-Scan, Double Tap = Add to Total)
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
        forceScanAndSpeak();
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
    if (el.modeBanknoteBtn) {
      el.modeBanknoteBtn.addEventListener("click", () => switchDetectionMode("banknote"));
    }
    if (el.modeCoinBtn) {
      el.modeCoinBtn.addEventListener("click", () => switchDetectionMode("coin"));
    }

    if (el.tapArea) {
      el.tapArea.addEventListener("click", (e) => {
        if (e.target.closest(".nav-circle-btn, .floating-zoom-btn, .pill-torch-btn, .menu-bottom-sheet")) return;
        handleScreenTap();
      });
    }

    if (el.btnTorch) {
      el.btnTorch.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleTorch();
      });
    }

    if (el.btnZoom) {
      el.btnZoom.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleZoom();
      });
    }

    if (el.btnSwitchCam) {
      el.btnSwitchCam.addEventListener("click", (e) => {
        e.stopPropagation();
        switchCamera();
      });
    }

    if (el.btnFixCam) {
      el.btnFixCam.addEventListener("click", async (e) => {
        e.stopPropagation();
        getar(40);
        speak("Menghubungkan langsung ke webcam...");
        await switchCamera();
      });
    }

    if (el.btnMenu) {
      el.btnMenu.addEventListener("click", (e) => {
        e.stopPropagation();
        openMenu();
      });
    }

    if (el.aiChip) {
      el.aiChip.addEventListener("click", (e) => {
        e.stopPropagation();
        openMenu();
      });
    }

    if (el.btnCloseSheet) {
      el.btnCloseSheet.addEventListener("click", closeMenu);
    }

    if (el.menuBackdrop) {
      el.menuBackdrop.addEventListener("click", (e) => {
        if (e.target === el.menuBackdrop) closeMenu();
      });
    }

    if (el.btnAddTotal) {
      el.btnAddTotal.addEventListener("click", addToTotal);
    }
    if (el.btnResetTotal) {
      el.btnResetTotal.addEventListener("click", resetTotal);
    }

    if (el.btnBack) {
      el.btnBack.addEventListener("click", (e) => {
        e.stopPropagation();
        forceScanAndSpeak();
      });
    }

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
  // Bootstrapping Aplikasi (Non-blocking & Auto-recovery)
  // -------------------------------------------------------------------------
  // -------------------------------------------------------------------------
  // Bootstrapping Aplikasi (Non-blocking & Auto-recovery)
  // -------------------------------------------------------------------------
  async function bootstrap() {
    attachEventListeners();

    // Jalankan inisialisasi AI model dan kamera secara paralel (non-blocking)
    initCoinModel().catch((err) => console.warn("[MLOps] Error model koin:", err));
    initTfjsModel().catch((err) => console.warn("[MLOps] Error model uang kertas:", err));
    startCamera().catch((err) => console.warn("[MLOps] Error inisialisasi kamera:", err));

    // Auto-recovery jika izin kamera baru saja diberikan oleh pengguna di popup browser
    if (navigator.permissions && navigator.permissions.query) {
      try {
        const perm = await navigator.permissions.query({ name: "camera" });
        perm.onchange = () => {
          console.log("[MLOps Camera] Status izin kamera berubah:", perm.state);
          if (perm.state === "granted") {
            startCamera(app.currentFacingMode);
          }
        };
      } catch (_) {}
    }

    // Auto-recovery saat tab browser kembali aktif (window focus atau visibilitychange)
    const handleReactivation = () => {
      if (!app.stream || (el.video && (el.video.paused || el.video.videoWidth === 0))) {
        console.log("[MLOps Camera] Jendela aktif kembali / visibilitas pulih, menyegarkan kamera...");
        startCamera(app.currentFacingMode);
      }
    };
    window.addEventListener("focus", handleReactivation);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") handleReactivation();
    });

    // Fallback interaksi: sentuh/klik di mana saja langsung menyalakan kamera
    const onUserInteraction = async () => {
      if (!app.stream || (el.video && (el.video.paused || el.video.videoWidth === 0))) {
        console.log("[MLOps Camera] Gestur pengguna diterima, menyalakan kamera...");
        await startCamera(app.currentFacingMode);
      } else if (el.video && el.video.paused) {
        try {
          el.video.muted = true;
          await el.video.play();
        } catch (err) {
          console.warn("[MLOps Camera] Gagal memutar video:", err);
        }
      }
    };

    document.addEventListener("pointerdown", onUserInteraction, { passive: true });
    document.addEventListener("click", onUserInteraction, { passive: true });

    setTimeout(() => {
      speak("Rupiah Reader siap. Deteksi otomatis aktif. Arahkan uang kertas atau uang koin ke kamera.");
    }, 600);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
