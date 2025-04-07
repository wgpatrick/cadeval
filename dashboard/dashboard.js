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
    const modelNames = Object.keys(metaStatistics);

    // Destroy existing charts if they exist
    Object.values(charts).forEach(chart => chart?.destroy());

    // Extract data for charts (including new ones)
    const successRates = modelNames.map(m => metaStatistics[m].overall_pipeline_success_rate);
    const checksPassedRates = modelNames.map(m => metaStatistics[m].all_geo_checks_passed_rate_rel);
    const avgChamfer = modelNames.map(m => metaStatistics[m].average_chamfer_distance); // Renamed field
    const avgHausdorff = modelNames.map(m => metaStatistics[m].average_hausdorff_99p_distance); // New
    const avgVolumeDiff = modelNames.map(m => metaStatistics[m].average_volume_diff_percent); // New
    const volumePassRates = modelNames.map(m => metaStatistics[m].volume_check_pass_rate_rel); // New
    const hausdorffPassRates = modelNames.map(m => metaStatistics[m].hausdorff_check_pass_rate_rel); // New

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
                        if (context.parsed.y !== null) {
                            // Add % for rate charts
                            const suffix = context.dataset.label.includes('Rate') ? '%' : '';
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

     const percentDiffChartOptions = { // Use specific name for % diff
          ...commonChartOptions,
         scales: { y: { beginAtZero: true, ticks: { callback: value => value + '%' } } } // Y-axis as percentage diff
      };


    // 1. Overall Success Rate Chart
    const successCtx = document.getElementById('successRateChart').getContext('2d');
    charts.successRate = new Chart(successCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Overall Success Rate',
                data: successRates,
                backgroundColor: 'rgba(40, 167, 69, 0.6)', // Success green
                borderColor: 'rgba(40, 167, 69, 1)',
                borderWidth: 1
            }]
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Overall Success Rate (% Tasks Fully Passed)' }}} // Add title
    });

    // 2. Checks Passed Rate Chart
    const checksCtx = document.getElementById('checksPassedChart').getContext('2d');
    charts.checksPassed = new Chart(checksCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Metric Checks Passed Rate (Rel. to Checks Run)', // Updated label
                // Replace nulls with NaN for Chart.js to skip them gracefully
                 data: checksPassedRates.map(d => d === null ? NaN : d),
                backgroundColor: 'rgba(255, 193, 7, 0.6)', // Warning yellow
                borderColor: 'rgba(255, 193, 7, 1)',
                borderWidth: 1
            }]
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Metric Checks Passed Rate (% Rel. to Checks Run)' }}} // Updated title
    });

    // 3. Average Chamfer Distance Chart
    const chamferCtx = document.getElementById('avgSimilarityChart').getContext('2d');
     charts.similarity = new Chart(chamferCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Avg Chamfer Distance (mm)',
                // Replace nulls with NaN for Chart.js to skip them gracefully
                data: avgChamfer.map(d => d === null ? NaN : d),
                 backgroundColor: 'rgba(153, 102, 255, 0.6)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }]
        },
         options: { ...distanceChartOptions, plugins: { ...distanceChartOptions.plugins, title: { display: true, text: 'Avg Chamfer Distance (Lower is Better)' }}} // Add title
    });

    // 4. Average Hausdorff Distance Chart
    const hausdorffCtx = document.getElementById('avgHausdorffChart').getContext('2d');
    charts.avgHausdorff = new Chart(hausdorffCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Avg Hausdorff 99p Distance (mm)',
                // Replace nulls with NaN for Chart.js to skip them gracefully
                data: avgHausdorff.map(d => d === null ? NaN : d),
                 backgroundColor: 'rgba(23, 162, 184, 0.6)',
                borderColor: 'rgba(23, 162, 184, 1)',
                borderWidth: 1
            }]
        },
         options: { ...distanceChartOptions, plugins: { ...distanceChartOptions.plugins, title: { display: true, text: 'Avg Hausdorff 99p Distance (Lower is Better)' }}} // Add title
    });

    // 5. Average Volume Difference Chart
    const volumeDiffCtx = document.getElementById('avgVolumeDiffChart').getContext('2d');
    charts.avgVolumeDiff = new Chart(volumeDiffCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Avg Volume Difference (%)',
                // Replace nulls with NaN for Chart.js to skip them gracefully
                data: avgVolumeDiff.map(d => d === null ? NaN : d),
                 backgroundColor: 'rgba(253, 126, 20, 0.6)',
                borderColor: 'rgba(253, 126, 20, 1)',
                borderWidth: 1
            }]
        },
         options: { ...percentDiffChartOptions, plugins: { ...percentDiffChartOptions.plugins, title: { display: true, text: 'Avg Volume Difference (% vs Reference)' }}} // Add title
    });

    // 6. Volume Check Pass Rate Chart
    const volumePassCtx = document.getElementById('volumePassRateChart').getContext('2d');
    charts.volumePassRate = new Chart(volumePassCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Volume Check Pass Rate (%)',
                // Replace nulls with NaN for Chart.js to skip them gracefully
                data: volumePassRates.map(d => d === null ? NaN : d),
                 backgroundColor: 'rgba(111, 66, 193, 0.6)',
                borderColor: 'rgba(111, 66, 193, 1)',
                borderWidth: 1
            }]
        },
         options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Volume Check Pass Rate (% Rel. to Checks Run)' }}} // Add title
    });

    // 7. Hausdorff Check Pass Rate Chart
    const hausdorffPassCtx = document.getElementById('hausdorffPassRateChart').getContext('2d');
    charts.hausdorffPassRate = new Chart(hausdorffPassCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Hausdorff Check Pass Rate (%)',
                // Replace nulls with NaN for Chart.js to skip them gracefully
                data: hausdorffPassRates.map(d => d === null ? NaN : d),
                 backgroundColor: 'rgba(214, 51, 132, 0.6)',
                borderColor: 'rgba(214, 51, 132, 1)',
                borderWidth: 1
            }]
        },
         options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Hausdorff Check Pass Rate (% Rel. to Checks Run)' }}} // Add title
    });

    const chartsContainer = document.getElementById('charts-container');
    if (chartsContainer) {
        chartsContainer.style.display = 'flex'; // Show charts container
    } else {
        console.error("Element with ID 'charts-container' not found when trying to display charts.");
    }
}


// --- HTML Table Creation ---
const SIMILARITY_THRESHOLD_MM = 1.0; // Chamfer threshold (matches default in geometry_check)
const BOUNDING_BOX_TOLERANCE_MM = 0.5; // BBox threshold (matches value in user's config.yaml)
const HAUSDORFF_THRESHOLD_MM = 0.5; // Hausdorff threshold (matches geometry_check.py)
const VOLUME_THRESHOLD_PERCENT = 1.0; // Volume threshold (matches geometry_check.py)

function createModelHtmlTable(modelName, modelResults, container) {
    const modelHeader = document.createElement('h2');
    modelHeader.textContent = `Model: ${modelName}`;
    container.appendChild(modelHeader);

    const table = document.createElement('table');
    table.className = 'results-table'; // Add class for styling
    const thead = table.createTHead();
    const tbody = table.createTBody();

    // Define columns - adapt these as needed from your data structure
    const columns = [
        { key: 'task_id', header: 'Task ID' },
        { key: 'scad_generation_success', header: 'SCAD Gen', format: formatBooleanText, isBoolean: true },
        { key: 'render_success', header: 'Render OK', format: formatBooleanText, isBoolean: true }, // Use 'check_render_successful'
        // { key: 'render_status_detail', header: 'Render Status' }, // Can be added back if needed
        { key: 'geometry_check_run_success', header: 'Checks Run', format: formatBooleanText, isBoolean: true },
        //{ key: 'individual_geometry_checks_passed', header: 'Metric Checks Passed', format: formatBooleanText, isBoolean: true }, // Maybe redundant with individual cols?

        // Add individual check columns dynamically if they exist
        ...(modelResults[0]?.individual_check_statuses ?
            Object.keys(modelResults[0].individual_check_statuses)
              // Filter to only include the core checks we want to display individually
              .filter(key => [
                  'check_render_successful', // Already covered by 'Render OK'?
                  'check_is_watertight',
                  'check_is_single_component',
                  'check_bounding_box_accurate',
                  'check_volume_passed', // New
                  'check_hausdorff_passed' // New
                ].includes(key))
              .map(checkKey => ({
                key: `individual_check_statuses.${checkKey}`,
                // Nicer Header logic
                header: checkKey === 'check_is_single_component' ? 'Component Count' :
                        checkKey === 'check_bounding_box_accurate' ? `BBox Acc. (< ${BOUNDING_BOX_TOLERANCE_MM.toFixed(1)}mm)` :
                        checkKey === 'check_volume_passed' ? `Volume Passed (< ${VOLUME_THRESHOLD_PERCENT.toFixed(1)}% Diff)` :
                        checkKey === 'check_hausdorff_passed' ? `Hausdorff Passed (< ${HAUSDORFF_THRESHOLD_MM.toFixed(1)} mm)` :
                        checkKey.replace('check_', '')
                                .replace(/_/g, ' ')
                                .replace('is ', '')
                                .replace('accurate', 'Acc.')
                                .replace('successful', 'OK')
                                .split(' ')
                                .map(w => w.charAt(0).toUpperCase() + w.slice(1))
                                .join(' '),
                format: formatPassFailText, // Use the Pass/Fail formatter for display text
                isBoolean: true // Mark as boolean for styling/logic
            })) : []),
        // Added Chamfer Passed column
        { key: 'chamfer_check_passed', header: `Chamfer Passed (< ${SIMILARITY_THRESHOLD_MM.toFixed(1)}mm)`, format: formatPassFailText, isBoolean: true },
        // Removed isSimilarity: true from Chamfer distance
        { key: 'geometric_similarity_distance', header: 'Chamfer (mm)', format: formatSimilarityText, threshold: SIMILARITY_THRESHOLD_MM }, 
        { key: 'hausdorff_99p_distance_detail', header: 'Hausdorff (mm)', format: formatDetailText, isMetric: true }, // Display detail string
        { key: 'volume_reference_detail', header: 'Vol Ref (mm³)', format: formatDetailText, isMetric: true }, // Display detail string
        { key: 'volume_generated_detail', header: 'Vol Gen (mm³)', format: formatDetailText, isMetric: true }, // Display detail string
        { key: 'bbox_reference_detail', header: 'BBox Ref (mm)', format: formatDetailText, isMetric: true }, // Display detail string
        { key: 'bbox_generated_aligned_detail', header: 'BBox Gen Aligned (mm)', format: formatDetailText, isMetric: true }, // Display detail string

        // New column for visualization command
        { key: 'visualize_cmd', header: 'Visualize Cmd' },

        // { key: 'geometry_check_error_detail', header: 'Check Error' }, // Can be added back if needed
        // { key: 'generation_error', header: 'Generation Error' } // Can be added back if needed
    ];

    // Create header row
    const headerRow = thead.insertRow();
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.header;

        headerRow.appendChild(th);
    });

    // Create data rows
    modelResults.forEach(result => {
        const row = tbody.insertRow();
        let isRowFullySuccessful = true; // Assume success initially for the row
        let taskIdCell = null; // To store the Task ID cell for later styling

        columns.forEach(col => {
            const cell = row.insertCell();
            // Handle nested keys like 'individual_check_statuses.check_manifold'
            let rawValue = col.key.split('.').reduce((obj, key) => obj?.[key], result);
            let displayValue;
            let cellStatusClass = ''; // e.g., 'status-yes', 'status-no', 'status-na'

            // --- Handle specific columns ---
            if (col.key === 'visualize_cmd') {
                const refPath = result.reference_stl_path;
                const genPath = result.output_stl_path;
                // Only create command if both paths exist
                if (refPath && genPath) {
                    const task = result.task_id || 'task';
                    const model = result.model_name || 'model';
                    const title = `Viz: ${task}-${model} (Ref=Green, Gen=Red)`;
                    // Construct command - assuming paths are relative to project root
                    displayValue = `python scripts/visualize_comparison.py --ref "${refPath}" --gen "${genPath}" --title "${title}"`;
                    cell.textContent = displayValue;
                } else {
                    displayValue = 'N/A'; // Cannot visualize if paths missing
                    cell.textContent = displayValue;
                    cellStatusClass = 'status-na';
                }
            } else {
                 // Handle nested keys like 'individual_check_statuses.check_manifold'
                 let rawValue = col.key.split('.').reduce((obj, key) => obj?.[key], result);
    
                // Handle formatting for other columns
                 if (col.format) { 
                     displayValue = col.format(rawValue);
                     cell.textContent = displayValue;
                 } else {
                     displayValue = rawValue !== null && typeof rawValue !== 'undefined' ? rawValue : 'N/A';
                     cell.textContent = displayValue;
                 }
            }

            // Determine cell status class for styling
            if (displayValue === 'N/A' || rawValue === null || typeof rawValue === 'undefined') {
                cellStatusClass = 'status-na';
                // N/A in certain critical columns means overall failure
                if ([ 'scad_generation_success',
                     'render_success',
                     'geometry_check_run_success',
                     `individual_check_statuses.check_is_watertight`,
                     `individual_check_statuses.check_is_single_component`,
                     `individual_check_statuses.check_bounding_box_accurate`,
                     `individual_check_statuses.check_volume_passed`, // New
                     `individual_check_statuses.check_hausdorff_passed` // New
                    ].includes(col.key))
                {
                     isRowFullySuccessful = false;
                }
            } else if (col.isBoolean) {
                 if (rawValue === true) {
                      cellStatusClass = 'status-yes';
                  } else {
                      cellStatusClass = 'status-no';
                      // If this boolean check is required for success, mark row as failed
                      if ([ 'scad_generation_success',
                           'render_success',
                           `individual_check_statuses.check_is_watertight`,
                           `individual_check_statuses.check_is_single_component`,
                           `individual_check_statuses.check_bounding_box_accurate`,
                           `individual_check_statuses.check_volume_passed`, // New
                           `individual_check_statuses.check_hausdorff_passed` // New
                         ].includes(col.key)) {
                           isRowFullySuccessful = false;
                      }
                  }
             } else if (col.isMetric) {
                 // Apply general styling but don't affect row success based on metric value itself
                 // (Success determined by the boolean check columns)
                 cellStatusClass = 'status-metric'; // Class for styling metric values if needed
             }

            // Apply the status class to the current cell
            if (cellStatusClass) { 
                cell.classList.add(cellStatusClass);
            }

            // Store the Task ID cell when we encounter it
            if (col.key === 'task_id') {
                taskIdCell = cell;
            }
        });

        // After processing all cells in the row, style the Task ID cell based on overall row success
        if (taskIdCell) {
            taskIdCell.classList.add(isRowFullySuccessful ? 'status-yes' : 'status-no');
        }
    });

    container.appendChild(table);
}


// --- Dashboard Initialization (fetches data automatically) ---
async function initializeDashboard() {
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorIndicator = document.getElementById('error-indicator');
    const chartsContainer = document.getElementById('charts-container');
    const gridsContainer = document.getElementById('grids-container');

    // Show loading indicator initially, clear errors/content
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    if (errorIndicator) errorIndicator.style.display = 'none';
    if (chartsContainer) chartsContainer.style.display = 'none';
    if (gridsContainer) gridsContainer.innerHTML = '';

    try {
        // Fetch data (Restored)
        console.log("Fetching dashboard_data.json...");
        const response = await fetch('./dashboard_data.json');
        console.log("Fetch response status:", response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} - Could not fetch dashboard_data.json. Make sure it exists in the dashboard/ directory and you've run scripts/process_results.py.`);
        }
        const data = await response.json();
        console.log("Data loaded successfully:", data);

        // Validate data structure (basic checks)
        if (!data || typeof data !== 'object') {
             throw new Error("Invalid data format: Input is not an object.");
         }
         if (!data.results_by_model || typeof data.results_by_model !== 'object') {
             throw new Error("Invalid data format: Missing or invalid 'results_by_model'.");
         }
          if (!data.meta_statistics || typeof data.meta_statistics !== 'object') {
             throw new Error("Invalid data format: Missing or invalid 'meta_statistics'.");
         }


        // Populate header
        const runIdElement = document.getElementById('run-id');
        if (runIdElement) {
            // Use run_id from fetched data
             runIdElement.textContent = `CadEval Dashboard - Run: ${data.run_id || 'Unknown'}`;
         } else {
             console.error("Element with ID 'run-id' not found.");
         }

        // Render charts
        if (data.meta_statistics && Object.keys(data.meta_statistics).length > 0) {
            console.log("Rendering summary charts...");
            renderSummaryCharts(data.meta_statistics);
            // Ensure container is shown after rendering
            if (chartsContainer) chartsContainer.style.display = 'flex';
        } else {
            console.warn("No meta-statistics found in data.");
            if (chartsContainer) {
                 chartsContainer.innerHTML = '<p style="text-align:center; width:100%;">No summary statistics available.</p>';
                 chartsContainer.style.display = 'block'; // Show the message
            } else {
                 console.error("Element with ID 'charts-container' not found.");
             }
        }

        // Create tables
        if (gridsContainer && data.results_by_model && Object.keys(data.results_by_model).length > 0) {
            console.log("Creating results tables...");
             for (const [modelName, modelResults] of Object.entries(data.results_by_model)) {
                 // Basic check if modelResults is an array before processing
                 if (Array.isArray(modelResults)) {
                     createModelHtmlTable(modelName, modelResults, gridsContainer);
                 } else {
                      console.warn(`Skipping table for model ${modelName}: results are not an array.`);
                 }
             }
         } else if (!gridsContainer) {
             console.error("Element with ID 'grids-container' not found.");
        } else {
             console.warn("No results_by_model found in data or container missing.");
             if (gridsContainer) gridsContainer.innerHTML = '<p style="text-align:center; width:100%;">No detailed results available.</p>'; // Show message
        }

        // Hide loading indicator on success
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        console.log("Dashboard initialized successfully.");

    } catch (error) {
        console.error('Error initializing dashboard:', error);
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        if (errorIndicator) {
            errorIndicator.textContent = `Error initializing dashboard: ${error.message}`;
            errorIndicator.style.display = 'block';
        } else {
             console.error("Element with ID 'error-indicator' not found.");
         }
         // Keep charts/grids containers hidden on error
         if (chartsContainer) chartsContainer.style.display = 'none';
         if (gridsContainer) gridsContainer.innerHTML = ''; // Clear grids container on error too
    }
}

// --- Automatically Initialize on DOM Ready ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed. Initializing dashboard...");
    initializeDashboard(); // Call initialization automatically
});

// NOTE: Removed the file input handling logic.
// Initialization now happens automatically by fetching dashboard_data.json. 