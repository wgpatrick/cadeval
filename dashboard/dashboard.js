let charts = {
    successRate: null,
    checksPassed: null,
    similarity: null
};

// --- Helper Functions ---
function formatBooleanStatus(value) {
    if (value === true) {
        return { icon: 'success', text: 'Yes' };
    } else if (value === false) {
        return { icon: 'failure', text: 'No' };
    } else {
        return { icon: 'na', text: 'N/A' }; // For null or undefined
    }
}

function formatSimilarity(value) {
    if (value === null || typeof value === 'undefined') {
        return 'N/A';
    }
    // Ensure value is treated as a number for comparison
    const numValue = Number(value);
    if (isNaN(numValue)) {
        return 'Invalid'; // Should ideally not happen if data is clean
    }
    // Format to 2 decimal places
    return numValue.toFixed(2) + ' mm';
}

// --- Cell Renderers for AG-Grid ---
function booleanStatusCellRenderer(params) {
    const status = formatBooleanStatus(params.value);
    return `<div class="cell-status-container"><span class="status-indicator status-${status.icon}"></span><span>${status.text}</span></div>`;
}

function individualCheckCellRenderer(params) {
    const status = formatBooleanStatus(params.value);
    return `<div class="cell-status-container"><span class="status-indicator status-${status.icon}"></span></div>`; // Only show icon for brevity
}

function similarityCellRenderer(params) {
    const formattedValue = formatSimilarity(params.value);
    return `<span>${formattedValue}</span>`;
}


// --- Chart Creation ---
function renderSummaryCharts(metaStatistics) {
    const modelNames = Object.keys(metaStatistics);

    // Destroy existing charts if they exist
    Object.values(charts).forEach(chart => chart?.destroy());

    // Extract data for charts
    const successRates = modelNames.map(m => metaStatistics[m].overall_success_rate * 100); // Convert to percentage
    const checksPassedRates = modelNames.map(m => metaStatistics[m].avg_checks_passed_rate * 100);
    const avgSimilarity = modelNames.map(m => metaStatistics[m].avg_similarity_distance_mm);

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

     const similarityChartOptions = {
         ...commonChartOptions,
        scales: { y: { beginAtZero: true } }
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
                label: 'Checks Passed Rate',
                // Replace nulls with NaN for Chart.js to skip them gracefully
                 data: checksPassedRates.map(d => d === null ? NaN : d),
                backgroundColor: 'rgba(255, 193, 7, 0.6)', // Warning yellow
                borderColor: 'rgba(255, 193, 7, 1)',
                borderWidth: 1
            }]
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Avg Geometry Checks Passed Rate (%)' }}} // Add title
    });

    // 3. Average Similarity Chart
    const similarityCtx = document.getElementById('avgSimilarityChart').getContext('2d');
     charts.similarity = new Chart(similarityCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Avg Similarity Distance (mm)',
                // Replace nulls with NaN for Chart.js to skip them gracefully
                data: avgSimilarity.map(d => d === null ? NaN : d),
                 backgroundColor: 'rgba(153, 102, 255, 0.6)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }]
        },
         options: { ...similarityChartOptions, plugins: { ...similarityChartOptions.plugins, title: { display: true, text: 'Avg Similarity Distance (Lower is Better)' }}} // Add title
    });

     document.getElementById('charts-container').style.display = 'flex'; // Show charts
}


// --- AG-Grid Creation ---
function createModelGrids(resultsByModel) {
    const gridsContainer = document.getElementById('grids-container');
    gridsContainer.innerHTML = ''; // Clear previous grids

    const defaultColDef = {
        sortable: true,
        filter: true,
        resizable: true,
         minWidth: 100,
         cellStyle: { 'line-height': '25px' } // Adjust for icon vertical centering if needed
    };

    const columnDefs = [
         { headerName: "Task ID", field: "task_id", pinned: 'left', width: 100, filter: 'agTextColumnFilter' },
        { headerName: "SCAD Gen", field: "scad_generation_success", cellRenderer: booleanStatusCellRenderer, width: 100 },
         { headerName: "Render", field: "render_success", cellRenderer: booleanStatusCellRenderer, width: 100 },
        { headerName: "Render Status", field: "render_status_detail", width: 150, hide: true }, // Hidden by default
         { headerName: "Checks Run", field: "geometry_check_orchestration_success", cellRenderer: booleanStatusCellRenderer, width: 100 },
         { headerName: "Checks Passed", field: "individual_geometry_checks_passed", cellRenderer: booleanStatusCellRenderer, width: 100 },
         // Dynamically add individual check columns
         ...(resultsByModel[Object.keys(resultsByModel)[0]]?.[0]?.individual_check_statuses ? // Check if statuses exist
              Object.keys(resultsByModel[Object.keys(resultsByModel)[0]][0].individual_check_statuses).map(checkKey => ({
                  headerName: checkKey.replace('check_', '').replace(/_/g, ' ').replace('accurate', 'Acc.').replace('successful','OK').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '), // Nicer Header
                  field: `individual_check_statuses.${checkKey}`,
                  cellRenderer: individualCheckCellRenderer,
                  width: 120
              })) : []),
         { headerName: "Similarity (mm)", field: "geometric_similarity_distance", cellRenderer: similarityCellRenderer, valueGetter: p => p.data.geometric_similarity_distance, width: 150 },
         { headerName: "Check Error", field: "geometry_check_error_detail", width: 200, hide: true }, // Hidden by default
         { headerName: "Generation Error", field: "generation_error", width: 200, hide: true }, // Hidden by default
    ];


    for (const [modelName, modelResults] of Object.entries(resultsByModel)) {
        const modelHeader = document.createElement('h2');
        modelHeader.textContent = `Model: ${modelName}`;
        gridsContainer.appendChild(modelHeader);

        const gridDiv = document.createElement('div');
        gridDiv.id = `grid-${modelName.replace(/[^a-zA-Z0-9]/g, '-')}`; // Create safe ID
        gridDiv.className = 'ag-theme-alpine';
        gridsContainer.appendChild(gridDiv);

        const gridOptions = {
            columnDefs: columnDefs,
            rowData: modelResults,
            defaultColDef: defaultColDef,
            domLayout: 'autoHeight', // Grid height adjusts to content
             pagination: true,
             paginationPageSize: 10, // Show 10 tasks per page
        };

        new agGrid.Grid(gridDiv, gridOptions);
    }
}

// --- Dashboard Initialization (called after AG-Grid is ready) ---
async function initializeDashboard() {
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorIndicator = document.getElementById('error-indicator');
    const chartsContainer = document.getElementById('charts-container');
    const gridsContainer = document.getElementById('grids-container');

    // Show loading indicator initially
    loadingIndicator.style.display = 'block';
    errorIndicator.style.display = 'none';
    chartsContainer.style.display = 'none';
    gridsContainer.innerHTML = '';

    try {
        // Fetch data
        const response = await fetch('./dashboard_data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} - Could not fetch dashboard_data.json. Make sure it's in the same directory as dashboard.html and that you've run scripts/process_results.py.`);
        }
        const data = await response.json();

        // Populate header
        document.getElementById('run-id').textContent = `CadEval Dashboard - Run: ${data.run_id || 'Unknown'}`;

        // Render charts
        if (data.meta_statistics && Object.keys(data.meta_statistics).length > 0) {
            renderSummaryCharts(data.meta_statistics);
        } else {
            console.warn("No meta-statistics found in data.");
            chartsContainer.innerHTML = '<p style="text-align:center; width:100%;">No summary statistics available.</p>';
            chartsContainer.style.display = 'block';
        }

        // Check AG-Grid again (as a final safeguard, though waitForAgGrid should ensure it)
        if (typeof agGrid === 'undefined' || typeof agGrid.Grid === 'undefined') {
           throw new Error("AG-Grid failed to become available.");
        }

        // Create grids
        if (data.results_by_model && Object.keys(data.results_by_model).length > 0) {
            createModelGrids(data.results_by_model);
        } else {
            console.warn("No detailed results found in data.");
            gridsContainer.innerHTML = '<p>No detailed model results available.</p>';
        }

    } catch (error) {
        console.error('Error initializing dashboard:', error);
        errorIndicator.textContent = `Error initializing dashboard: ${error.message}`;
        errorIndicator.style.display = 'block';
    } finally {
        // Hide loading indicator regardless of success or failure
        loadingIndicator.style.display = 'none';
    }
}

// --- Wait for AG-Grid to load ---
function waitForAgGrid(callback, timeout = 10000, interval = 100) {
    const startTime = Date.now();
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorIndicator = document.getElementById('error-indicator');

    loadingIndicator.style.display = 'block'; // Show loading indicator
    errorIndicator.style.display = 'none';

    const intervalId = setInterval(() => {
        if (typeof window.agGrid !== 'undefined' && typeof window.agGrid.Grid !== 'undefined') {
            console.log("AG-Grid is ready.");
            clearInterval(intervalId);
            callback(); // Run the main initialization logic
        } else if (Date.now() - startTime > timeout) {
            console.error("AG-Grid loading timed out.");
            clearInterval(intervalId);
            loadingIndicator.style.display = 'none';
            errorIndicator.textContent = 'Error: Failed to load AG-Grid library within timeout. Check network or CDN link.';
            errorIndicator.style.display = 'block';
        }
    }, interval);
}

// Start waiting for AG-Grid and then initialize
waitForAgGrid(initializeDashboard); 