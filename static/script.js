"use strict";

let currentData = null;
let originalImageSrc = null;

// --- 1. CLINICAL KNOWLEDGE BASE ---
const MEDICAL_CODES = {
    'fracture': {
        finding: "Cortical disruption noted with trabecular compression. Acute loss of vertebral height.",
        impression: "ACUTE VERTEBRAL COMPRESSION FRACTURE.",
        icd_code: "M48.50XA"
    },
    'misalignment': {
        finding: "Loss of standard spinal curvature. Translational displacement >3mm observed.",
        impression: "SPONDYLOLISTHESIS / SPINAL INSTABILITY.",
        icd_code: "M43.10"
    },
    'degeneration': {
        finding: "Reduced disc height, marginal osteophytes, and endplate sclerosis.",
        impression: "DEGENERATIVE DISC DISEASE (SPONDYLOSIS).",
        icd_code: "M47.812"
    },
    'normal': {
        finding: "Vertebral alignment preserved. Disc spaces maintained. No acute abnormalities.",
        impression: "UNREMARKABLE STUDY. NORMAL SPINAL ALIGNMENT.",
        icd_code: "Z00.00"
    }
};

const TREATMENT_PLANS = {
    'fracture': {
        dos: ["Strict bed rest for 48 hours", "Use prescribed back brace when moving", "Increase Calcium & Vitamin D intake"],
        donts: ["Do NOT lift heavy objects", "Avoid twisting or bending at the waist", "No high-impact activities"],
        therapy: "Gradual mobilization after 2 weeks. Isometric core strengthening only after pain subsides.",
        medication: "Analgesics (Acetaminophen), Calcitonin (for pain), Bisphosphonates (if osteoporotic)."
    },
    'misalignment': {
        dos: ["Maintain neutral spine posture", "Engage core before moving", "Sleep with a pillow under knees"],
        donts: ["Avoid hyperextension (arching back)", "No heavy lifting overhead", "Avoid prolonged standing"],
        therapy: "Physical Therapy focused on Core Stabilization (Williams Flexion Exercises). Hamstring stretching.",
        medication: "NSAIDs (Ibuprofen/Naproxen) for inflammation. Muscle relaxants for spasms."
    },
    'degeneration': {
        dos: ["Maintain healthy weight", "Use ergonomic chairs/standing desks", "Apply heat for stiffness, ice for pain"],
        donts: ["Avoid prolonged sitting (>1 hour)", "Avoid high-impact running", "Do not slouch"],
        therapy: "Low-impact aerobics (Swimming/Walking). Manual therapy for joint mobilization.",
        medication: "NSAIDs, Glucosamine/Chondroitin supplements. Epidural steroid injections if radicular."
    }
};

// --- HANDLERS ---
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    originalImageSrc = URL.createObjectURL(file);
    document.getElementById("imageViewer").innerHTML = `<img src="${originalImageSrc}" id="displayedImage">`;
    document.getElementById("resultsContent").innerHTML = "<p class='empty-state'>Analyzing Dicom Data...<br>Neural Ensemble Active...</p>";
    document.getElementById("viewControls").classList.add("hidden");
    document.getElementById("actionFooter").classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);

    fetch('/predict', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            currentData = data;
            displayResults(data);
        })
        .catch(err => {
            document.getElementById("resultsContent").innerHTML = `<p style='color:red'>Error: ${err.message}</p>`;
        });
}

function displayResults(data) {
    const medInfo = MEDICAL_CODES[data.predicted_class.toLowerCase()] || MEDICAL_CODES['normal'];
    const confidence = (data.confidence * 100).toFixed(1);

    const html = `
        <div class="diagnosis-box">
            <h2 class="diagnosis-title">${data.predicted_class.toUpperCase()}</h2>
            <span class="confidence-badge">AI Confidence: ${confidence}%</span>
            <span style="float:right; font-size:0.8rem; color:#666">ICD-10: ${medInfo.icd_code}</span>
        </div>

        <div class="graph-container">
            <img src="${data.graph_image}" alt="Probability Graph">
        </div>

        <div class="med-section">
            <h5>Radiographic Findings</h5>
            <p class="med-text">${medInfo.finding}</p>
        </div>

        <div class="med-section">
            <h5>Impression</h5>
            <p class="med-text" style="font-weight:bold">${medInfo.impression}</p>
        </div>

        <div class="med-section">
            <h5>Recommendations</h5>
            <p class="med-text" style="white-space: pre-line">${(medInfo.recommendation || "")}</p>
        </div>
    `;

    document.getElementById("resultsContent").innerHTML = html;
    document.getElementById("viewControls").classList.remove("hidden");
    document.getElementById("actionFooter").classList.remove("hidden");
}

function toggleView(type) {
    const img = document.getElementById("displayedImage");
    const btns = document.querySelectorAll('.btn-toggle');
    btns.forEach(b => b.classList.remove('active'));

    if (type === 'original') {
        img.src = originalImageSrc;
        btns[0].classList.add('active');
    } else {
        img.src = currentData.heatmap_image;
        btns[1].classList.add('active');
    }
}

// --- PROFESSIONAL REPORT GENERATOR ---
function generatePDF() {
    if (!currentData) return;
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    const pClass = currentData.predicted_class.toLowerCase();
    const medInfo = MEDICAL_CODES[pClass];
    const treatment = TREATMENT_PLANS[pClass];
    const today = new Date().toLocaleDateString();

    // HEADER
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.text("NEUROSPINE DIAGNOSTIC CENTER", 105, 15, null, null, "center");
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text("123 Medical Plaza, Radiology Dept | Phone: (555) 019-2834", 105, 22, null, null, "center");
    doc.line(10, 25, 200, 25);

    // PATIENT INFO
    doc.text(`Patient ID: ${Math.random().toString().substr(2, 8)}`, 15, 35);
    doc.text(`Date: ${today}`, 150, 35);
    doc.text(`Modality: DIGITAL X-RAY`, 150, 40);

    let yPos = 50;

    // FINDINGS & IMPRESSION
    doc.setFont("helvetica", "bold");
    doc.text("FINDINGS:", 15, yPos);
    doc.setFont("helvetica", "normal");
    doc.text(doc.splitTextToSize(medInfo.finding, 180), 15, yPos + 7);
    
    yPos += 25;
    doc.setFont("helvetica", "bold");
    doc.text("IMPRESSION:", 15, yPos);
    doc.setTextColor(192, 57, 43); // Red
    doc.text(medInfo.impression, 15, yPos + 7);
    doc.setTextColor(0);

    // IMAGES & GRAPH
    yPos += 20;
    try {
        // Heatmap Left
        doc.addImage(currentData.heatmap_image, 'PNG', 15, yPos, 80, 80);
        doc.setFontSize(8);
        doc.text("Fig 1. AI Saliency Map", 55, yPos + 85, null, null, "center");

        // Graph Right
        doc.addImage(currentData.graph_image, 'PNG', 110, yPos + 10, 80, 50);
        doc.text("Fig 2. Differential Probability", 150, yPos + 65, null, null, "center");
    } catch(e) { console.log("Image error"); }

    yPos += 100;

    // TREATMENT PLAN (Only if NOT Normal)
    if (pClass !== 'normal' && treatment) {
        doc.setFontSize(11);
        doc.setFont("helvetica", "bold");
        doc.text("RECOMMENDED TREATMENT PLAN:", 15, yPos);
        yPos += 10;

        doc.setFontSize(10);
        doc.setTextColor(0, 100, 0); // Green
        doc.text("DO'S:", 15, yPos);
        doc.setTextColor(0);
        doc.setFont("helvetica", "normal");
        treatment.dos.forEach(t => {
            doc.text(`• ${t}`, 20, yPos + 5);
            yPos += 5;
        });

        yPos += 5;
        doc.setFont("helvetica", "bold");
        doc.setTextColor(150, 0, 0); // Red
        doc.text("DON'TS:", 15, yPos);
        doc.setTextColor(0);
        doc.setFont("helvetica", "normal");
        treatment.donts.forEach(t => {
            doc.text(`• ${t}`, 20, yPos + 5);
            yPos += 5;
        });

        yPos += 10;
        doc.setFont("helvetica", "bold");
        doc.text("Therapy & Medication:", 15, yPos);
        doc.setFont("helvetica", "normal");
        doc.text(doc.splitTextToSize(treatment.therapy, 180), 15, yPos + 5);
        doc.text(doc.splitTextToSize(treatment.medication, 180), 15, yPos + 10);
    }

    // FOOTER
    doc.setTextColor(100);
    doc.setFont("helvetica", "italic");
    doc.setFontSize(8);
    doc.text("Electronically signed by AI Assistant. Verified by __________________", 105, 280, null, null, "center");

    doc.save(`Radiology_Report_${Date.now()}.pdf`);
}