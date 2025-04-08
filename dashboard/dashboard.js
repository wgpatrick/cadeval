let charts = {
    successRate: null,
    checksPassed: null,
    similarity: null,
    avgHausdorff: null,
    avgVolumeDiff: null,
    volumePassRate: null,
    hausdorffPassRate: null
};

// --- Helper Functions ---
// Simplified boolean formatting for table cells
function formatBooleanText(value) {
    if (value === true) {
        return 'Yes';
    } else if (value === false) {
        return 'No';
    } else {
        return 'N/A';
    }
}

// New formatter for Pass/Fail text
function formatPassFailText(value) {
    if (value === true) {
        return 'Pass';
    } else if (value === false) {
        return 'Fail';
    } else {
        return 'N/A';
    }
}

// Simplified similarity formatting for table cells
function formatSimilarityText(value) {
    if (value === null || typeof value === 'undefined') {
        return 'N/A';
    }
    const numValue = Number(value);
    if (isNaN(numValue)) {
        return 'Invalid';
    }
    return numValue.toFixed(2) + ' mm';
}

// New formatter for volume/bbox details
function formatDetailText(value) {
    return (value !== null && typeof value !== 'undefined') ? value : 'N/A';
}

// --- Cell Renderers for AG-Grid ---
function booleanStatusCellRenderer(params) {
    const status = formatBooleanText(params.value);
    return `<div class="cell-status-container"><span class="status-indicator status-${status.toLowerCase()}"></span><span>${status}</span></div>`;
}

function individualCheckCellRenderer(params) {
    const status = formatBooleanText(params.value);
    return `<div class="cell-status-container"><span class="status-indicator status-${status.toLowerCase()}"></span></div>`; // Only show icon for brevity
}

function similarityCellRenderer(params) {
    const formattedValue = formatSimilarityText(params.value);
    return `<span>${formattedValue}</span>`;
}


// --- Chart Creation ---
function renderSummaryCharts(metaStatistics) {
    if (!metaStatistics || Object.keys(metaStatistics).length === 0) {
        console.warn("No metaStatistics provided for charts.");
        document.getElementById('charts-container').style.display = 'none'; // Hide chart container
        // Optionally display a message
        const chartsContainer = document.getElementById('charts-container');
        let msgElement = document.getElementById('no-charts-msg');
        if (!msgElement) {
            msgElement = document.createElement('p');
            msgElement.id = 'no-charts-msg';
            msgElement.textContent = 'No summary statistics available to generate charts.';
            chartsContainer.parentNode.insertBefore(msgElement, chartsContainer);
        } else {
            msgElement.style.display = 'block';
        }
        return;
    }

    // Hide no-charts message if it exists
    const noChartsMsg = document.getElementById('no-charts-msg');
    if (noChartsMsg) noChartsMsg.style.display = 'none';

    // Show chart container
    document.getElementById('charts-container').style.display = 'grid'; // Use grid as per HTML class

    const modelNames = Object.keys(metaStatistics);

    // Destroy existing charts if they exist
    Object.values(charts).forEach(chart => chart?.destroy());

    // Extract data for charts using NEW field names
    const successRates = modelNames.map(m => metaStatistics[m].overall_pass_rate); // Use overall_pass_rate
    const chamferPassRates = modelNames.map(m => metaStatistics[m].chamfer_pass_rate); // Use chamfer_pass_rate for checks passed chart for now
    const avgChamfer = modelNames.map(m => metaStatistics[m].avg_chamfer); // Use avg_chamfer
    const avgHausdorff95p = modelNames.map(m => metaStatistics[m].avg_hausdorff_95p); // Use avg_hausdorff_95p
    // Skipping avgVolumeDiff chart for now as data isn't directly calculated
    const volumePassRates = modelNames.map(m => metaStatistics[m].volume_pass_rate);
    const hausdorffPassRates = modelNames.map(m => metaStatistics[m].hausdorff_pass_rate);

    const commonChartOptions = {
        scales: {
            y: { beginAtZero: true }
        },
        maintainAspectRatio: false, // Allow chart to resize within wrapper
        plugins: {
             legend: { display: false }, // Hide legend if only one dataset
             tooltip: {
                 callbacks: {
                    label: function(context) {
                        let label = context.dataset.label || '';
                        if (label) {
                             label += ': ';
                        }
                        if (context.parsed.y !== null && !isNaN(context.parsed.y)) {
                            // Add % for rate charts, mm for distance
                            let suffix = '';
                            if (context.dataset.label.includes('Rate')) {
                                suffix = '%';
                            } else if (context.dataset.label.includes('(mm)')) {
                                suffix = ' mm';
                            }
                            label += context.parsed.y.toFixed(1) + suffix;
                        }
                        return label;
                    }
                 }
             }
        }
    };

    const rateChartOptions = {
         ...commonChartOptions,
        scales: { y: { beginAtZero: true, max: 100, ticks: { callback: value => value + '%' } } } // Y-axis as percentage
    };

     const distanceChartOptions = { // Use specific name for distance
         ...commonChartOptions,
        scales: { y: { beginAtZero: true } }
     };

     // Removed percentDiffChartOptions as chart is skipped


    // 1. Overall Success Rate Chart (Canvas ID: successRateChart)
    const successCtx = document.getElementById('successRateChart').getContext('2d');
    charts.successRate = new Chart(successCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Overall Pass Rate', // Updated Label
                data: successRates.map(d => d === null ? NaN : d), // Handle potential nulls
                backgroundColor: 'rgba(40, 167, 69, 0.6)', // Success green
                borderColor: 'rgba(40, 167, 69, 1)',
                borderWidth: 1
            }]
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Overall Pass Rate (%)' }}} // Updated Title
    });

    // 2. Chamfer Pass Rate Chart (Using checksPassedChart canvas ID for now)
    const checksCtx = document.getElementById('checksPassedChart').getContext('2d');
    charts.checksPassed = new Chart(checksCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Chamfer Pass Rate', // Updated Label
                 data: chamferPassRates.map(d => d === null ? NaN : d),
                backgroundColor: 'rgba(255, 193, 7, 0.6)', // Warning yellow
                borderColor: 'rgba(255, 193, 7, 1)',
                borderWidth: 1
            }]
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Chamfer Pass Rate (% Rel. to Checks Run)' }}} // Updated title
    });

    // 3. Average Chamfer Distance Chart (Canvas ID: avgSimilarityChart)
    const chamferCtx = document.getElementById('avgSimilarityChart').getContext('2d');
     charts.similarity = new Chart(chamferCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Avg Chamfer Distance (mm)', // Label ok
                data: avgChamfer.map(d => d === null ? NaN : d),
                 backgroundColor: 'rgba(153, 102, 255, 0.6)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }]
        },
         options: { ...distanceChartOptions, plugins: { ...distanceChartOptions.plugins, title: { display: true, text: 'Avg Chamfer Distance (mm, Lower is Better)' }}} // Updated title
    });

    // 4. Average Hausdorff 95p Distance Chart (Canvas ID: avgHausdorffChart)
    const hausdorffCtx = document.getElementById('avgHausdorffChart').getContext('2d');
    charts.avgHausdorff = new Chart(hausdorffCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Avg Hausdorff 95p Distance (mm)', // Updated Label (95p)
                data: avgHausdorff95p.map(d => d === null ? NaN : d),
                 backgroundColor: 'rgba(23, 162, 184, 0.6)',
                borderColor: 'rgba(23, 162, 184, 1)',
                borderWidth: 1
            }]
        },
         options: { ...distanceChartOptions, plugins: { ...distanceChartOptions.plugins, title: { display: true, text: 'Avg Hausdorff 95p Distance (mm, Lower is Better)' }}} // Updated title
    });

    // 5. Skipping Volume Diff Chart (Canvas ID: avgVolumeDiffChart)
    // We can hide this canvas or reuse it later if needed.
    const avgVolumeDiffCanvas = document.getElementById('avgVolumeDiffChart');
    if (avgVolumeDiffCanvas && avgVolumeDiffCanvas.parentElement) {
         avgVolumeDiffCanvas.parentElement.style.display = 'none'; // Hide the wrapper div
    }

    // 6. Volume Pass Rate Chart (Canvas ID: volumePassRateChart)
    const volumePassCtx = document.getElementById('volumePassRateChart').getContext('2d');
    charts.volumePassRate = new Chart(volumePassCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Volume Pass Rate', // New Label
                data: volumePassRates.map(d => d === null ? NaN : d),
                backgroundColor: 'rgba(255, 99, 132, 0.6)', // Example color
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Volume Pass Rate (% Rel. to Checks Run)' }}} // Add title
    });

    // 7. Hausdorff Pass Rate Chart (Canvas ID: hausdorffPassRateChart)
    const hausdorffPassCtx = document.getElementById('hausdorffPassRateChart').getContext('2d');
    charts.hausdorffPassRate = new Chart(hausdorffPassCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Hausdorff Pass Rate', // New Label
                data: hausdorffPassRates.map(d => d === null ? NaN : d),
                backgroundColor: 'rgba(54, 162, 235, 0.6)', // Example color
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Hausdorff Pass Rate (% Rel. to Checks Run)' }}} // Add title
    });

}


// --- NEW: Render Summary Tables ---
function renderSummaryTables(metaStatistics, taskStatistics) {
    const container = document.getElementById('summary-tables-container');
    if (!container) {
        console.error("Summary tables container not found!");
        return;
    }
    container.innerHTML = ''; // Clear previous content

    // Helper to format numbers for tables
    const fmtTable = (val, suffix = '', precision = 1) => {
        if (val === null || typeof val === 'undefined' || isNaN(Number(val))) {
            return 'N/A';
        }
        return Number(val).toFixed(precision) + suffix;
    };

    // -- Model Summary Table --
    if (metaStatistics && Object.keys(metaStatistics).length > 0) {
        const modelTable = document.createElement('table');
        modelTable.className = 'summary-table'; // Add class for styling
        const modelCaption = modelTable.createCaption();
        modelCaption.textContent = 'Model Performance Summary';

        const modelHeader = modelTable.createTHead().insertRow();
        const modelHeaders = ['Model', 'Overall Pass (%)', 'SCAD Gen (%)', 'Render (%)', 'Checks Run', 'Chamfer Pass (%)', 'Haus. Pass (%)', 'Vol. Pass (%)', 'Avg Chamfer (mm)', 'Avg Haus. 95p (mm)'];
        modelHeaders.forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            modelHeader.appendChild(th);
        });

        const modelBody = modelTable.createTBody();
        for (const [modelName, stats] of Object.entries(metaStatistics)) {
            const row = modelBody.insertRow();
            row.insertCell().textContent = modelName;
            row.insertCell().textContent = fmtTable(stats.overall_pass_rate, '%');
            row.insertCell().textContent = fmtTable(stats.scad_generation_success_rate, '%');
            row.insertCell().textContent = fmtTable(stats.render_success_rate, '%');
            row.insertCell().textContent = stats.checks_run_count ?? 'N/A'; // Direct count
            row.insertCell().textContent = fmtTable(stats.chamfer_pass_rate, '%');
            row.insertCell().textContent = fmtTable(stats.hausdorff_pass_rate, '%');
            row.insertCell().textContent = fmtTable(stats.volume_pass_rate, '%');
            row.insertCell().textContent = fmtTable(stats.avg_chamfer, ' mm', 2);
            row.insertCell().textContent = fmtTable(stats.avg_hausdorff_95p, ' mm', 2);
        }
        container.appendChild(modelTable);
    } else {
        container.innerHTML += '<p>No model summary statistics available.</p>';
    }

    // -- Task Summary Table --
    if (taskStatistics && Object.keys(taskStatistics).length > 0) {
        const taskTable = document.createElement('table');
        taskTable.className = 'summary-table'; // Add class for styling
        const taskCaption = taskTable.createCaption();
        taskCaption.textContent = 'Task Performance Summary (Across Models)';
        taskTable.style.marginTop = '20px'; // Add some space between tables

        const taskHeader = taskTable.createTHead().insertRow();
        const taskHeaders = ['Task ID', 'Overall Pass (%)', 'SCAD Gen (%)', 'Render (%)', 'Checks Run', 'Chamfer Pass (%)', 'Haus. Pass (%)', 'Vol. Pass (%)', 'Avg Chamfer (mm)', 'Avg Haus. 95p (mm)'];
        taskHeaders.forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            taskHeader.appendChild(th);
        });

        const taskBody = taskTable.createTBody();
        // Sort tasks for consistent order (optional)
        const sortedTaskIds = Object.keys(taskStatistics).sort(); 
        for (const taskId of sortedTaskIds) {
            const stats = taskStatistics[taskId];
            const row = taskBody.insertRow();
            row.insertCell().textContent = taskId;
            row.insertCell().textContent = fmtTable(stats.overall_pass_rate, '%');
            row.insertCell().textContent = fmtTable(stats.scad_generation_success_rate, '%');
            row.insertCell().textContent = fmtTable(stats.render_success_rate, '%');
            row.insertCell().textContent = stats.checks_run_count ?? 'N/A'; // Direct count
            row.insertCell().textContent = fmtTable(stats.chamfer_pass_rate, '%');
            row.insertCell().textContent = fmtTable(stats.hausdorff_pass_rate, '%');
            row.insertCell().textContent = fmtTable(stats.volume_pass_rate, '%');
            row.insertCell().textContent = fmtTable(stats.avg_chamfer, ' mm', 2);
            row.insertCell().textContent = fmtTable(stats.avg_hausdorff_95p, ' mm', 2);
        }
        container.appendChild(taskTable);
    } else {
         if (!metaStatistics || Object.keys(metaStatistics).length === 0) {
             // Only add task message if model message wasn't already added
             container.innerHTML += '<p>No task summary statistics available.</p>';
         } else {
             // Add space if model table exists but task table doesn't
             const spacer = document.createElement('div');
             spacer.style.height = '20px';
             container.appendChild(spacer);
         }
    }
}
// --- END: Render Summary Tables ---

// --- HTML Table Creation ---
const SIMILARITY_THRESHOLD_MM = 1.0; // Chamfer threshold (matches default in geometry_check)
const BOUNDING_BOX_TOLERANCE_MM = 0.5; // BBox threshold (matches value in user's config.yaml)
const HAUSDORFF_THRESHOLD_MM = 0.5; // Hausdorff threshold (matches geometry_check.py)
const VOLUME_THRESHOLD_PERCENT = 1.0; // Volume threshold (matches geometry_check.py)

function createModelHtmlTable(modelName, modelResults, container) {
    const table = document.createElement('table');
    table.classList.add('results-table');
    const thead = table.createTHead();
    const tbody = table.createTBody();

    // Define Columns (aligned with NEW keys from process_results.py)
    const columns = [
        { key: 'task_id', header: 'Task ID' },
        { key: 'replicate_id', header: 'Rep ID' }, // NEW Replicate ID column
        { key: 'scad_generation_success', header: 'SCAD Gen', format: formatPassFailText, isBoolean: true }, // Use new key
        { key: 'check_render_successful', header: 'Render OK', format: formatPassFailText, isBoolean: true }, // Use refined key
        { key: 'checks_run_attempted', header: 'Checks Run', format: formatPassFailText, isBoolean: true }, // Use new key
        { key: 'check_is_watertight', header: 'Watertight', format: formatPassFailText, isBoolean: true },
        { key: 'check_is_single_component', header: 'Single Comp', format: formatPassFailText, isBoolean: true },
        { key: 'check_bounding_box_accurate', header: 'BBox Acc.', format: formatPassFailText, isBoolean: true, tooltip: "Checks if aligned BBox dims are within tolerance" }, // Use refined key
        { key: 'check_volume_passed', header: 'Volume Pass', format: formatPassFailText, isBoolean: true, tooltip: "Checks if volume difference % is within threshold" }, // Use refined key
        { key: 'check_hausdorff_passed', header: 'Hausdorff Pass', format: formatPassFailText, isBoolean: true, tooltip: "Checks if Hausdorff 95p distance is within threshold" }, // Use refined key
        { key: 'chamfer_check_passed', header: 'Chamfer Pass', format: formatPassFailText, isBoolean: true, tooltip: "Checks if Chamfer distance is within threshold" }, // Use new key

        { key: 'chamfer_dist', header: 'Chamfer (mm)' }, // Already formatted string
        {
            key: 'haus_95p_dist',
            header: 'Hausdorff Dist (95p / 99p mm)',
            hozAlign: "center",
            headerHozAlign: "center",
            // Custom handling moved inside the loop
        },
        { key: 'ref_vol', header: 'Vol Ref (mm³)' }, // Already formatted string
        { key: 'gen_vol', header: 'Vol Gen (mm³)' }, // Already formatted string
        { key: 'ref_bbox', header: 'BBox Ref (mm)' }, // Display raw array or N/A
        { key: 'gen_bbox', header: 'BBox Gen Aligned (mm)' }, // Display raw array or N/A
        { key: 'visualize_cmd', header: 'Visualize Cmd' },
        // Error columns (optional)
        // { key: 'gen_err', header: 'Gen Error' },
        // { key: 'render_err', header: 'Render Error' },
        // { key: 'check_err', header: 'Check Error' },
    ];

    // Create header row
    const headerRow = thead.insertRow();
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.header;
        if (col.tooltip) {
             th.title = col.tooltip;
        }
        headerRow.appendChild(th);
    });

    // Create data rows
    modelResults.forEach(result => {
        const row = tbody.insertRow();
        // Use the pre-calculated overall_passed flag from the data
        const isRowFullySuccessful = result.overall_passed === true; 

        let taskIdCell = null; // To store the Task ID cell for later styling
        let repIdCell = null; // To store the Rep ID cell

        columns.forEach(col => {
            const cell = row.insertCell();
            let rawValue = result[col.key]; // Direct access using the key
            let displayValue = rawValue; // Default display value
            let cellStatusClass = '';

            // --- Specific Column Handling ---
            if (col.key === 'visualize_cmd') {
                const refPath = result.ref_stl_path;
                const genPath = result.stl_path;
                if (refPath && genPath) {
                    const task = result.task_id || 'task';
                    const model = result.model_name || 'model';
                    const rep = result.replicate_id || 'repX';
                    const title = `Viz: ${task}-${model}-Rep${rep} (Ref=Green, Gen=Red)`;
                    displayValue = `python scripts/visualize_comparison.py --ref \"${refPath}\" --gen \"${genPath}\" --title \"${title}\"`;
                    cell.textContent = "Run Cmd"; // Keep cell content short
                    cell.title = displayValue; // Put full command in tooltip
                    cell.style.cursor = 'pointer'; // Indicate clickable
                    cell.onclick = () => navigator.clipboard.writeText(displayValue).then(() => alert('Command copied to clipboard!'), () => alert('Failed to copy command.'));
                } else {
                    displayValue = 'N/A';
                    cell.textContent = displayValue;
                }
            } else if (col.key === 'haus_95p_dist') {
                 const value95p = result.haus_95p_dist || 'N/A'; // Use pre-formatted value
                 const value99p = result.haus_99p_dist || 'N/A';
                 displayValue = `95p: ${value95p}<br>99p: ${value99p}`;
                 if (value95p === 'N/A' && value99p === 'N/A') {
                     cell.style.fontStyle = "italic";
                     cell.style.color = "#888";
                     displayValue = 'N/A';
                     cell.innerHTML = displayValue;
                 } else {
                    cell.innerHTML = displayValue; // Use innerHTML for <br>
                 }
                 cell.style.whiteSpace = 'normal';
                 cell.style.textAlign = 'center';
            } else if (col.key === 'ref_bbox' || col.key === 'gen_bbox') {
                 // Format BBox arrays
                 if (Array.isArray(rawValue)) {
                      displayValue = `[${rawValue.map(v => parseFloat(v).toFixed(1)).join(', ')}]`;
                      cell.textContent = displayValue;
                 } else {
                      cell.textContent = 'N/A';
                 }
            } else {
                 // --- General Formatting and Styling ---
                 // Use formatter if defined
                 if (col.format) {
                     displayValue = col.format(rawValue);
                 } else {
                     // Use raw value, handle null/undefined
                     displayValue = (rawValue !== null && typeof rawValue !== 'undefined') ? rawValue : 'N/A';
                 }
                 cell.textContent = displayValue;
            }

            // --- Determine Cell Status Class ---
            // Check for null, undefined, OR the specific string "N/A"
            if (rawValue === null || typeof rawValue === 'undefined' || displayValue === 'N/A') {
                cellStatusClass = 'status-na';
            } else if (col.isBoolean) {
                cellStatusClass = rawValue === true ? 'status-yes' : 'status-no';
            }
            // Add other specific styling logic if needed (e.g., for metrics)

            // Apply the status class
            if (cellStatusClass) {
                cell.classList.add(cellStatusClass);
            }

            // Store specific cells for later row-level styling
            if (col.key === 'task_id') taskIdCell = cell;
            if (col.key === 'replicate_id') repIdCell = cell;
        });

        // Style Task ID cell based on overall success
        if (taskIdCell) {
            taskIdCell.classList.add(isRowFullySuccessful ? 'status-yes' : 'status-no');
        }
         // Optionally add styling to Rep ID cell too
         if (repIdCell) {
             // Example: Add subtle background based on success
             repIdCell.style.backgroundColor = isRowFullySuccessful ? '#e6ffed' : '#ffebee'; 
         }
    });

    container.appendChild(table);
}


// --- Dashboard Initialization (fetch data, render) ---
async function initializeDashboard() {
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorIndicator = document.getElementById('error-indicator');
    const gridsContainer = document.getElementById('grids-container');
    const chartsContainer = document.getElementById('charts-container');
    const runIdElement = document.getElementById('run-id');
    const summaryContainer = document.getElementById('summary-tables-container'); // Get summary container

    loadingIndicator.style.display = 'block';
    errorIndicator.style.display = 'none';
    gridsContainer.innerHTML = ''; // Clear previous grids
    chartsContainer.style.display = 'none'; // Hide charts initially
    summaryContainer.innerHTML = ''; // Clear previous summary tables

    try {
        // Fetch the processed data
        const response = await fetch('dashboard_data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Update Run ID title
        if (data.run_id) {
            runIdElement.textContent = `CadEval Dashboard - Run: ${data.run_id}`;
        }

        // Check if data exists
        if (!data || !data.results_by_model) { // Base check
             throw new Error('No results_by_model data found or data format is incorrect in dashboard_data.json');
        }

        // Render Summary Tables using meta_statistics and task_statistics
        renderSummaryTables(data.meta_statistics || {}, data.task_statistics || {});

        // Render Charts using meta_statistics
        renderSummaryCharts(data.meta_statistics || {}); // Pass meta_statistics, fallback to empty obj

        // Render Grids for each model
        if (Object.keys(data.results_by_model).length > 0) {
            for (const [modelName, modelResults] of Object.entries(data.results_by_model)) {
                createModelHtmlTable(modelName, modelResults, gridsContainer);
            }
        } else {
            gridsContainer.innerHTML = '<p>No model results found in the data.</p>';
        }

        loadingIndicator.style.display = 'none';

    } catch (error) {
        console.error('Error initializing dashboard:', error);
        loadingIndicator.style.display = 'none';
        errorIndicator.textContent = `Error loading dashboard data: ${error.message}. Please check console and dashboard_data.json.`;
        errorIndicator.style.display = 'block';
        chartsContainer.style.display = 'none'; // Ensure charts are hidden on error
        summaryContainer.innerHTML = '<p>Failed to load dashboard summary.</p>'; // Add error to summary area too
    }
}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', initializeDashboard);

// NOTE: Removed the file input handling logic.
// Initialization now happens automatically by fetching dashboard_data.json. 