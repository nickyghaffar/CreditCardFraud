// State and references
let currentFeatures = {};
let isStreaming = false;
let streamInterval = null;

// Initialize when DOM loads
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  buildRemainingFeaturesUI();
  setupSliderSync();
  loadLeaderboard();
  
  // Run initial prediction with default values
  analyzeTransaction();
});

// Setup tab navigation
function setupTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      
      const tabId = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      document.getElementById(`tab-${tabId}`).classList.add("active");
    });
  });
}

// Generate UI for features V1 to V28 (excluding key ones V3, V4, V10, V12, V14, V17)
function buildRemainingFeaturesUI() {
  const container = document.getElementById("other-features-container");
  const keyFeatures = ["V3", "V4", "V10", "V12", "V14", "V17"];
  
  for (let i = 1; i <= 28; i++) {
    const fName = `V${i}`;
    if (keyFeatures.includes(fName)) continue;
    
    const div = document.createElement("div");
    div.className = "acc-input";
    div.innerHTML = `
      <label>${fName}:</label>
      <input type="number" id="input-${fName.toLowerCase()}" step="0.1" value="0.0">
    `;
    container.appendChild(div);
  }
}

// Synchronize sliders with text values
function setupSliderSync() {
  const sliders = ["v14", "v10", "v4", "v17", "v12", "v3"];
  sliders.forEach(id => {
    const slider = document.getElementById(`input-${id}`);
    const display = document.getElementById(`val-${id}`);
    if (slider && display) {
      slider.addEventListener("input", (e) => {
        display.textContent = parseFloat(e.target.value).toFixed(2);
      });
    }
  });
}

// Collect all feature values from UI
function collectFeatures() {
  const features = {
    model_name: document.getElementById("model-select").value,
    Amount: parseFloat(document.getElementById("input-amount").value) || 0.0,
    Time: parseFloat(document.getElementById("input-time").value) || 0.0,
  };
  
  for (let i = 1; i <= 28; i++) {
    const fName = `V${i}`;
    const el = document.getElementById(`input-${fName.toLowerCase()}`);
    features[fName] = el ? (parseFloat(el.value) || 0.0) : 0.0;
  }
  
  return features;
}

// Set form values from an object
function populateForm(data) {
  if (data.Amount !== undefined) document.getElementById("input-amount").value = data.Amount;
  if (data.Time !== undefined) document.getElementById("input-time").value = data.Time;
  
  for (let i = 1; i <= 28; i++) {
    const fName = `V${i}`;
    const id = `input-${fName.toLowerCase()}`;
    const el = document.getElementById(id);
    if (el && data[fName] !== undefined) {
      el.value = parseFloat(data[fName]).toFixed(2);
      // also update slider display if exists
      const valDisp = document.getElementById(`val-${fName.toLowerCase()}`);
      if (valDisp) {
        valDisp.textContent = parseFloat(data[fName]).toFixed(2);
      }
    }
  }
}

// Load Presets from API
async function loadPreset(type) {
  try {
    const res = await fetch(`/api/sample/${type}`);
    const data = await res.json();
    populateForm(data.features);
    
    // Auto trigger prediction
    setTimeout(analyzeTransaction, 100);
  } catch (err) {
    console.error("Failed to load preset:", err);
  }
}

// Analyze transaction using selected AI model
async function analyzeTransaction() {
  const btn = document.getElementById("btn-analyze");
  btn.style.opacity = "0.7";
  btn.innerHTML = `<span class="btn-icon">⏳</span> در حال پردازش در شبکه عصبی...`;
  
  const payload = collectFeatures();
  
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    const result = await res.json();
    displayResults(result);
  } catch (err) {
    console.error("Prediction error:", err);
  } finally {
    btn.style.opacity = "1";
    btn.innerHTML = `<span class="btn-icon">⚡</span> تحلیل هوشمند و استعلام ریسک`;
  }
}

// Render Results & Animate Gauge
function displayResults(res) {
  const prob = res.fraud_probability;
  const percent = res.fraud_percentage;
  
  // Update SVG arc gauge
  const fillArc = document.getElementById("gauge-fill-arc");
  const text = document.getElementById("gauge-text");
  
  // Total arc circumference for R=90 semi-circle is 282.74
  const totalLength = 282.74;
  const clampedProb = Math.min(1.0, Math.max(0, prob));
  const offset = totalLength * (1.0 - clampedProb);
  
  if (fillArc) {
    fillArc.style.strokeDashoffset = offset;
  }
  if (text) {
    text.textContent = `${percent.toFixed(1)}٪`;
    if (prob >= 0.50) {
      text.style.color = "var(--color-danger)";
    } else if (prob >= 0.20) {
      text.style.color = "var(--color-warning)";
    } else {
      text.style.color = "var(--color-success)";
    }
  }
  
  // Update decision box
  const box = document.getElementById("decision-box");
  const icon = document.getElementById("decision-icon");
  const title = document.getElementById("decision-title");
  const action = document.getElementById("decision-action");
  const badgeStatus = document.getElementById("badge-status");
  const actionDesc = document.getElementById("action-desc");
  
  box.className = "decision-box";
  
  if (prob >= 0.50) {
    box.classList.add("critical");
    icon.textContent = "🚨";
    title.textContent = "هشدار بحرانی: کلاهبرداری شناسایی شد!";
    action.textContent = "کارت بلافاصله مسدود شده و به کاربر پیامک ارسال شود.";
    badgeStatus.textContent = "کلاهبرداری قطعی";
    badgeStatus.className = "card-badge danger";
    actionDesc.textContent = "دستور اتوماتیک: تراکنش رد شد (Error 96 - Fraud Block). پرونده جهت پیگیری قضایی در صف شاپرک ثبت شد.";
  } else if (prob >= 0.20) {
    box.style.background = "rgba(255, 179, 0, 0.12)";
    box.style.borderColor = "rgba(255, 179, 0, 0.4)";
    icon.textContent = "⚠️";
    title.textContent = "تراکنش مشکوک به ناهنجاری";
    action.textContent = "ارسال رمز یکبار مصرف جدید (OTP) و احراز هویت دوعاملی.";
    badgeStatus.textContent = "نیازمند تایید پیامکی";
    badgeStatus.className = "card-badge warning";
    actionDesc.textContent = "دستور اتوماتیک: تراکنش در حالت انتظار (Pending) قرار گرفت تا مشتری کد تایید دومرحله‌ای را وارد کند.";
  } else {
    box.classList.add("safe");
    icon.textContent = "✅";
    title.textContent = "تراکنش عادی، امن و تاییدشده";
    action.textContent = "تسویه بانکی بلامانع است و اجازه انتقال داده شد.";
    badgeStatus.textContent = "امن و تاییدشده";
    badgeStatus.className = "card-badge success";
    actionDesc.textContent = "دستور اتوماتیک: تراکنش موفق (Approved 00). هیچ نشانه‌ای از الگوهای سرقت کارت دیده نشد.";
  }
  
  document.getElementById("res-model").textContent = res.model_used;
  document.getElementById("res-risk").textContent = res.risk_level;
  document.getElementById("res-valid").textContent = prob < 0.5 ? "معتبر (Authorized)" : "مسدود (Blocked)";
}

// Load Leaderboard Table
async function loadLeaderboard() {
  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();
    
    const tbody = document.getElementById("leaderboard-tbody");
    tbody.innerHTML = "";
    
    data.forEach((row, i) => {
      const tr = document.createElement("tr");
      const rankBadge = i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : `${i + 1}`;
      const prVal = (row["PR-AUC"] * 100).toFixed(2);
      const precVal = (row["Precision"] * 100).toFixed(1);
      const recVal = (row["Recall"] * 100).toFixed(1);
      const f1Val = row["F1-Score"].toFixed(4);
      
      tr.innerHTML = `
        <td><strong>${rankBadge}</strong></td>
        <td><strong>${row["Model"]}</strong></td>
        <td><span class="badge-tag success">${prVal}٪</span></td>
        <td><strong>${f1Val}</strong></td>
        <td>${precVal}٪</td>
        <td>${recVal}٪</td>
        <td>${row["Train Time (s)"].toFixed(1)} ثانیه</td>
        <td style="color: #ff3860;">${row["Missed (FN)"]}</td>
        <td style="color: #ffb300;">${row["False Alarms (FP)"]}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Leaderboard load failed:", err);
  }
}

// Live Stream Simulator
function toggleStream() {
  const btnText = document.getElementById("stream-btn-text");
  const icon = document.getElementById("stream-icon");
  
  if (!isStreaming) {
    isStreaming = true;
    btnText.textContent = "توقف استریم";
    icon.textContent = "⏸️";
    fetchAndPrependStreamBatch();
    streamInterval = setInterval(fetchAndPrependStreamBatch, 2500);
  } else {
    isStreaming = false;
    btnText.textContent = "شروع استریم زنده";
    icon.textContent = "▶️";
    clearInterval(streamInterval);
  }
}

async function fetchAndPrependStreamBatch() {
  try {
    const res = await fetch("/api/stream?count=3");
    const data = await res.json();
    const tbody = document.getElementById("stream-tbody");
    
    // Remove empty state if present
    const empty = tbody.querySelector(".empty-state");
    if (empty) empty.parentElement.remove();
    
    for (const tx of data.transactions) {
      // Evaluate each with selected model
      const evalRes = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: document.getElementById("model-select").value,
          ...tx.features
        })
      });
      const result = await evalRes.json();
      
      const tr = document.createElement("tr");
      const isFraud = result.is_fraud;
      tr.style.animation = "fadeIn 0.4s ease";
      
      tr.innerHTML = `
        <td><code>${tx.tx_id}</code></td>
        <td>$${tx.amount.toFixed(2)}</td>
        <td>${tx.time.toFixed(0)}</td>
        <td><span class="badge-tag ${tx.actual === 'FRAUD' ? 'danger' : 'success'}">${tx.actual}</span></td>
        <td><strong>${result.fraud_percentage.toFixed(1)}٪</strong></td>
        <td><span class="badge-tag ${isFraud ? 'danger' : result.fraud_probability > 0.2 ? 'warning' : 'success'}">${result.risk_level}</span></td>
        <td>${result.action}</td>
      `;
      
      tbody.insertBefore(tr, tbody.firstChild);
      
      // Keep only top 15 rows
      if (tbody.children.length > 15) {
        tbody.removeChild(tbody.lastChild);
      }
    }
  } catch (err) {
    console.error("Stream error:", err);
  }
}

function clearStreamTable() {
  const tbody = document.getElementById("stream-tbody");
  tbody.innerHTML = `<tr><td colspan="7" class="empty-state">جدول پاکسازی شد. دکمه «شروع استریم» را بزنید.</td></tr>`;
}

// Modal zoom functions
function openModal(src, caption) {
  const modal = document.getElementById("image-modal");
  const img = document.getElementById("modal-img");
  const cap = document.getElementById("modal-caption");
  modal.style.display = "flex";
  img.src = src;
  cap.textContent = caption;
}

function closeModal() {
  document.getElementById("image-modal").style.display = "none";
}
