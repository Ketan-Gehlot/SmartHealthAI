const diseaseSelect = document.getElementById("diseaseSelect");
const form = document.getElementById("predictionForm");

if (diseaseSelect) {
    diseaseSelect.addEventListener("change", generateForm);
}

function generateForm() {
    form.innerHTML = "";
    const disease = diseaseSelect.value;

    let fields = [];

    if (disease === "diabetes") {
        fields = ["glucose", "blood_pressure", "bmi", "age"];
    }
    else if (disease === "liver") {
        fields = ["age", "total_bilirubin", "direct_bilirubin", "alkaline_phosphotase", "alamine_aminotransferase", "aspartate_aminotransferase", "total_proteins", "albumin", "albumin_globulin_ratio"];
    }
    else if (disease === "cancer") {
        fields = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"];
    }
    else if (disease === "parkinsons") {
        fields.push("sequence (comma separated values)");
    }

    fields.forEach(f => {
        const label = document.createElement("label");
        label.innerText = f;

        const input = document.createElement("input");
        input.name = f;
        input.required = true;

        form.appendChild(label);
        form.appendChild(input);
    });
}

function submitPrediction() {
    const disease = diseaseSelect.value;
    const inputs = form.querySelectorAll("input");
    let data = {};

    inputs.forEach(i => {
        data[i.name] = parseFloat(i.value);
    });

    let endpoint = "/api/predict/" + disease;

    fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
        .then(res => res.json())
        .then(result => {
            document.getElementById("resultBox").innerHTML =
                `Result: ${result.result}<br>Confidence: ${result.confidence}%`;
        })
        .catch(() => {
            alert("Prediction failed. Check input values.");
        });
}
