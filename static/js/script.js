const vibDataForm = document.getElementById("vib-data-form");
const advancedL10Btn = document.getElementById('advanced-l10-btn');
const l10FormFieldset = document.getElementById('L10-form-fieldset');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const submitText = document.querySelector('.submit-text');
const submitLoading = document.querySelector('.submit-loading');

let l10FormAdvanced = false;
let vibPrediction = null;

function createNominalDegradationData() {
    const points = 100;
    const labels = [];
    const values = [];
    
    for (let i = 0; i <= points; i++) {
        labels.push(i);
        // Exponential degradation curve
        values.push((1 - (i / 100) ** 2) * 100);
    }
    
    return { labels, values };
}

function createActualHealthData(currentHealth) {
    const points = 100;
    const data = [];
    
    for (let i = 0; i <= points; i++) {
        // Simulate degradation with some noise
        const base = currentHealth;
        const degradation = (i / points) * (100 - base);
        const noise = Math.sin(i / 10) * 3;
        data.push(Math.max(0, Math.min(100, base - degradation + noise)));
    }
    
    return data;
}

function showResults(data) {
    // Hide error section
    errorSection.style.display = 'none';
    resultsSection.innerHTML = '';

    const rawMse = data['mse-raw']
    const mse = data['mse']

    Object.entries(data).forEach(([bearingName, bearingObject]) => {
        
        const bearingData = data[bearingName]
        const bearingStatus = getHealthClass(bearingData.health_score);
        const bearingCard = document.createElement('div')
        
        if (bearingName.startsWith("bearing_")) {
            const healthChartElement = document.createElement('canvas')
            healthChartElement.id = `health-chart-${bearingName}`;
            healthChartElement.style.maxHeight = '300px';



            bearingCard.className = 'bearing-card'
            bearingCard.innerHTML = `
                <h3>${bearingName.replace('_', ' ').toUpperCase()}</h3>
                <div class="bearing-metrics">
                    <p>Health Score: <span class="health-score">${bearingData.health_score}</span></p>
                    <p>Status: <span class="bearing-status">${bearingStatus}</span></p>
                </div>
            `;

            bearingCard.appendChild(healthChartElement)
            resultsSection.appendChild(bearingCard)

            let context = healthChartElement.getContext('2d');
            const degradationData = createNominalDegradationData()

            let actualHealthXValue = Math.sqrt(1 - (bearingData.health_score / 100)) * 100

            let healthChart = new Chart(context, {
                type: 'line',
                data: {
                    labels: degradationData.labels,
                    datasets: [
                        {
                            label: 'Theoretical Degradation',
                            data: degradationData.values,
                            borderColor: '#28a745',
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            fill: true,
                            tension: 0.3 
                        },
                        {
                            label: 'Current Status',
                            data: [{x: actualHealthXValue, y: [bearingData.health_score]}],  // Single point
                            borderColor: '#F46036',
                            backgroundColor: '#F46036',
                            pointRadius: 8,
                            pointHoverRadius: 10,
                            fill: false,
                            showLine: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,  // ← Fixed typo
                    scales: {
                        y: {
                            min: 0,
                            max: 100,
                            type: 'linear',
                            title: {
                                display: true,
                                text: 'Health Score (%)'
                            }
                        },
                        x: {
                            type: 'linear',
                            title: {
                                display: true,
                                text: 'Life Percentage (%)'
                            }
                        }
                    }
                }
            });
        } else {

        }
    })

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    function getHealthClass(healthScore) {
        if (healthScore >= 70) return 'Health-Good';
        if (healthScore >= 40) return 'Health-Moderate';
        if (healthScore >= 20) return 'Health-Low';
        return 'Health-Critical';
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(message = null) {
    // Hide results section
    resultsSection.style.display = 'none';
    
    // Update error message if provided
    if (message) {
        document.getElementById('error-message').textContent = message;
    }
    
    // Show error section
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth' });
}

function toggleL10AdvancedForm() {
    l10FormFieldset.innerHTML = `
        <div class="form-group">
            <label for="basic-rating">Basic Dynamic Load Rating (C)</label>
            <input type="number" id="basic-rating" name="basic-rating" 
                   step="0.01" min="0" placeholder="e.g. 12500">
            <small style="color: var(--color-text-muted); font-size: 0.85rem;">
                Load rating in Newtons (N)
            </small>
        </div>
        
        <div class="form-group">
            <label for="equivalent-load">Equivalent Dynamic Load (P)</label>
            <input type="number" id="equivalent-load" name="equivalent-load" 
                   step="0.01" min="0" placeholder="e.g. 2500">
            <small style="color: var(--color-text-muted); font-size: 0.85rem;">
                Applied load in Newtons (N)
            </small>
        </div>
        
        <div class="form-group">
            <label for="load-exponent">Load Exponent (p)</label>
            <input type="number" id="load-exponent" name="load-exponent" 
                   value="3" step="0.1" min="0" placeholder="3">
            <small style="color: var(--color-text-muted); font-size: 0.85rem;">
                Ball bearings: 3, Roller bearings: 10/3
            </small>
        </div>
    `;
    l10FormAdvanced = true;
    advancedL10Btn.textContent = "Use Simple L10 Input";
    advancedL10Btn.classList.remove('btn-outline');
    advancedL10Btn.classList.add('btn-secondary');
}

function toggleL10BasicForm() {
    l10FormFieldset.innerHTML = `
        <div class="form-group">
            <label for="vib-life-span">Rated Bearing Lifespan (L10)</label>
            <input type="number" id="vib-life-span" name="vib-life-span" 
                   step="1" min="0" placeholder="e.g. 1900000">
            <small style="color: var(--color-text-muted); font-size: 0.85rem;">
                Enter L10 rating in revolutions
            </small>
        </div>
    `;
    l10FormAdvanced = false;
    advancedL10Btn.textContent = "Advanced L10 Calculation";
    advancedL10Btn.classList.remove('btn-secondary');
    advancedL10Btn.classList.add('btn-outline');
}

function setLoadingState(loading) {
    if (loading) {
        submitText.style.display = 'none';
        submitLoading.style.display = 'inline-block';
        vibDataForm.classList.add('loading');
    } else {
        submitText.style.display = 'inline-block';
        submitLoading.style.display = 'none';
        vibDataForm.classList.remove('loading');
    }
}

// Event Listeners
advancedL10Btn.addEventListener('click', () => {
    if (!l10FormAdvanced) {
        toggleL10AdvancedForm();
    } else {
        toggleL10BasicForm();
    }
});

vibDataForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    
    // Hide any existing results/errors
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    // Set loading state
    setLoadingState(true);

    const formData = new FormData(event.target);

    try {
        const response = await fetch('/api/input', {
            method: 'POST',
            body: formData
        });

        const responseText = await response.text();
        console.log('Raw Response:', responseText);
        console.log('Response Status:', response.status);

        let result;
        try {
            result = JSON.parse(responseText);
        } catch (parseError) {
            console.error('JSON Parse Error:', parseError);
            console.error('Received text was:', responseText);
            showError('Server returned invalid response. Please try again.');
            return;
        }

        if (response.ok) {
            console.log('SUCCESS:', result);
            // TODO: Needs to be sensor_data
            vibPrediction = result;
            showResults(result);
        } else {
            console.error('ERROR:', result);
            showError(result.error || 'Analysis failed. Please check your input data.');
        }
        
    } catch (error) {
        console.error('Network Error:', error);
        showError('Network error occurred. Please check your connection and try again.');
    } finally {
        setLoadingState(false);
    }
});



