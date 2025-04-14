let charts = {
    successRate: null,
    renderSuccessRate: null,  // Moved up in the order
    checksRunSuccessRate: null, // New chart for checks run success rate
    checksPassed: null,
    similarity: null,
    avgHausdorff: null,
    avgVolumeDiff: null,
    volumePassRate: null,
    hausdorffPassRate: null,
    taskSuccessRate: null,
    complexityChart: null
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
    // Check for explicit true or the string "Pass" (case-insensitive)
    if (value === true || String(value).toLowerCase() === 'pass') {
        return 'Pass';
    // Check for explicit false or the string "Fail" (case-insensitive)
    } else if (value === false || String(value).toLowerCase() === 'fail') {
        return 'Fail';
    // Everything else (null, undefined, other strings) becomes N/A
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
function renderSummaryCharts(metaStatistics, taskStatistics, resultsByModel) {
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

    // Hide no-charts message if it exists and we have some data
    const noChartsMsg = document.getElementById('no-charts-msg');
    if (noChartsMsg && (metaStatistics || taskStatistics) && (Object.keys(metaStatistics || {}).length > 0 || Object.keys(taskStatistics || {}).length > 0) ) {
        noChartsMsg.style.display = 'none';
    }

    // Show chart container if we have any stats
    if ((metaStatistics && Object.keys(metaStatistics).length > 0) || (taskStatistics && Object.keys(taskStatistics).length > 0)) {
        document.getElementById('charts-container').style.display = 'grid'; // Use grid as per HTML class
    }

    // Destroy existing charts if they exist
    Object.values(charts).forEach(chart => chart?.destroy());

    // --- Data Preparation for Grouped Bars --- START ---
    const modelNames = Object.keys(metaStatistics);
    // Assume all models have the same set of prompt keys for simplicity
    // Get prompt keys from the first model
    const promptKeys = Object.keys(metaStatistics[modelNames[0]] || {}); 

    // --- Determine reliable total_count per prompt --- START ---
    const promptTotalCounts = {};
    promptKeys.forEach(promptKey => {
        let foundTotal = 0;
        // 1. Try finding in meta_statistics first
        for (const modelName of modelNames) {
            const total = metaStatistics[modelName]?.[promptKey]?.total_count;
            if (total !== null && total !== undefined && total > 0) {
                foundTotal = total;
                break; // Use the first valid non-zero total count found
            }
        }

        // 2. Fallback: Infer from results_by_model if not found in meta_statistics
        if (foundTotal === 0 && resultsByModel && modelNames.length > 0) { 
            const firstModelName = modelNames[0];
            const resultsForFirstModel = resultsByModel[firstModelName] || [];
            // Filter results for the current prompt key (case-insensitive just in case)
            const resultsForPrompt = resultsForFirstModel.filter(result => 
                result.Prompt && result.Prompt.toLowerCase() === promptKey.toLowerCase()
            );
            foundTotal = resultsForPrompt.length; // Count the actual results for this prompt
            if (foundTotal > 0) {
                console.log(`Inferred total_count=${foundTotal} for prompt '${promptKey}' from results_by_model.`);
            }
        }

        promptTotalCounts[promptKey] = foundTotal; // Store the determined total
        if (foundTotal === 0) {
            console.warn(`Could not determine reliable total_count > 0 for prompt '${promptKey}'. Tooltips might use 0 as denominator.`);
        }
    });
    console.log("Determined Prompt Total Counts (with fallback):", promptTotalCounts);
    // --- Determine reliable total_count per prompt --- END ---

    // Prepare datasets for each prompt key
    const datasets = {};
    promptKeys.forEach(promptKey => {
        datasets[promptKey] = {
            overallPassRate: [],
            renderSuccessRate: [], // Moved up
            checksRunSuccessRate: [], // New metric for checks run success rate
            chamferPassRate: [],
            avgChamfer: [],
            avgHausdorff95p: [],
            volumePassRate: [],
            hausdorffPassRate: [],
            modelIndices: [],
            providers: []
            // Add other metrics if needed
        };
    });

    // Populate data for each model and prompt
    modelNames.forEach((modelName, modelIndex) => {
        promptKeys.forEach(promptKey => {
            const stats = metaStatistics[modelName]?.[promptKey]; // Safe access
            let totalCount = stats?.total_count || 0; // Original total count from stats
            let renderSuccessCount = stats?.render_success_count || 0;
            let checksRunCount = stats?.checks_run_count || 0;
            let overallPassCount = stats?.overall_pass_count || 0; // Get the overall pass count
            
            // Use the reliable total count determined earlier for chart 2 denominator
            let reliableTotalCountForPrompt = promptTotalCounts[promptKey] || 0;

            datasets[promptKey].overallPassRate.push(stats?.overall_pass_rate ?? NaN);
            
            // --- Chart 2: STL Render Success Rate (%) --- START ---
            let chart2SuccessCount = renderSuccessCount;
            let chart2Value = stats?.render_success_rate ?? NaN;

            // Special handling for zoo-ml-text-to-cad
            if (modelName.includes('zoo-ml-text-to-cad')) {
                // If render_success_count is 0 but checks_run_count > 0, use checks_run_count
                if (chart2SuccessCount === 0 && checksRunCount > 0) {
                    console.log(`Correcting renderSuccessCount for ${modelName} (${promptKey}) using checksRunCount (${checksRunCount})`);
                    chart2SuccessCount = checksRunCount;
                    // Recalculate rate if possible
                    if (reliableTotalCountForPrompt > 0) {
                        chart2Value = (chart2SuccessCount / reliableTotalCountForPrompt) * 100;
                    } else {
                        chart2Value = NaN; // Cannot calculate rate without a valid total
                    }
                }
            }

            datasets[promptKey].renderSuccessRate.push({
                value: chart2Value,
                successCount: chart2SuccessCount, // Use potentially corrected count
                totalCount: reliableTotalCountForPrompt // Use the determined reliable count
            });
            // --- Chart 2: STL Render Success Rate (%) --- END ---
            
            // Chart 3: Success rate if STL rendered
            let checksSuccessRate = NaN;
            if (checksRunCount > 0) { // Calculate rate only if checks were run
                checksSuccessRate = (overallPassCount / checksRunCount) * 100;
            }

            datasets[promptKey].checksRunSuccessRate.push({
                value: checksSuccessRate,      // Rate based on checks run
                overallPassCount: overallPassCount, // Numerator for tooltip
                checksRunCount: checksRunCount    // Denominator for tooltip
            });
            
            datasets[promptKey].chamferPassRate.push(stats?.chamfer_pass_rate ?? NaN);
            datasets[promptKey].avgChamfer.push(stats?.avg_chamfer ?? NaN);
            datasets[promptKey].avgHausdorff95p.push(stats?.avg_hausdorff_95p ?? NaN);
            datasets[promptKey].volumePassRate.push(stats?.volume_pass_rate ?? NaN);
            datasets[promptKey].hausdorffPassRate.push(stats?.hausdorff_pass_rate ?? NaN);
            datasets[promptKey].modelIndices.push(modelIndex);
            datasets[promptKey].providers.push(modelName);
        });
    });
    
    // Function to sort model data by overall pass rate
    function sortModelDataBySuccessRate(promptKey) {
        // Get the data for this prompt
        const promptData = datasets[promptKey];
        const { overallPassRate, modelIndices } = promptData;
        
        // Create pairs of [passRate, index] for sorting
        const sortPairs = overallPassRate.map((rate, i) => [rate, i]);
        
        // Sort by pass rate (descending)
        sortPairs.sort((a, b) => b[0] - a[0]);
        
        // Create a sorted version of each array in the dataset
        const sortedDataset = {};
        for (const key of Object.keys(promptData)) {
            if (key !== 'modelIndices' && key !== 'providers') {
                sortedDataset[key] = sortPairs.map(pair => promptData[key][pair[1]]);
            }
        }
        
        // Also sort providers and model indices
        sortedDataset.modelIndices = sortPairs.map(pair => promptData.modelIndices[pair[1]]);
        sortedDataset.providers = sortPairs.map(pair => promptData.providers[pair[1]]);
        
        return sortedDataset;
    }

    // Function to create a special dataset for render success rate with detailed tooltips
    function createRenderSuccessRateDataset(promptKey, labelPrefix) {
        // Get the sorted data
        const sortedData = sortModelDataBySuccessRate(promptKey);
        
        return {
            label: `${labelPrefix} (${promptKey})`,
            // Extract the 'value' property from each object for display
            data: sortedData.renderSuccessRate.map(item => typeof item === 'object' ? item.value : item),
            backgroundColor: sortedData.providers.map(provider => {
                const key = getProvider(provider);
                return providerColors[key] || defaultColors[0];
            }),
            borderColor: sortedData.providers.map(provider => {
                const key = getProvider(provider);
                const color = providerColors[key] || defaultColors[0];
                return color.replace(', 0.6', ', 1');
            }),
            borderWidth: 1,
            // Store the complete objects for tooltip access
            _renderData: sortedData.renderSuccessRate,
            // Store original model indices for correct label mapping
            originalIndices: sortedData.modelIndices
        };
    }

    // Function to create the dataset for checks run success rate with detailed tooltips
    function createChecksRunSuccessRateDataset(promptKey, labelPrefix) {
        // Get the sorted data
        const sortedData = sortModelDataBySuccessRate(promptKey);
        
        return {
            label: `${labelPrefix} (${promptKey})`,
            // Extract the 'value' property from each object for display
            data: sortedData.checksRunSuccessRate.map(item => typeof item === 'object' ? item.value : item),
            backgroundColor: sortedData.providers.map(provider => {
                const key = getProvider(provider);
                return providerColors[key] || defaultColors[0];
            }),
            borderColor: sortedData.providers.map(provider => {
                const key = getProvider(provider);
                const color = providerColors[key] || defaultColors[0];
                return color.replace(', 0.6', ', 1');
            }),
            borderWidth: 1,
            // Store the complete objects for tooltip access
            _checksRunData: sortedData.checksRunSuccessRate,
            // Store original model indices for correct label mapping
            originalIndices: sortedData.modelIndices
        };
    }

    // Define provider-specific colors
    const providerColors = {
        'claude': 'rgba(54, 162, 235, 0.6)',   // Blue
        'gemini': 'rgba(255, 99, 132, 0.6)',   // Red
        'openai': 'rgba(75, 192, 192, 0.6)',   // Teal - All OpenAI models
        'gpt': 'rgba(75, 192, 192, 0.6)',      // Teal - OpenAI (fallback)
        'zoo': 'rgba(255, 206, 86, 0.6)',      // Yellow
        'chatgpt': 'rgba(75, 192, 192, 0.6)',  // Teal - OpenAI (fallback)
        'o1': 'rgba(75, 192, 192, 0.6)',       // Teal - OpenAI (fallback)
        'o3': 'rgba(75, 192, 192, 0.6)'        // Teal - OpenAI (fallback)
    };

    // Default colors for unknown providers
    const defaultColors = [
        'rgba(153, 102, 255, 0.6)', // Purple
        'rgba(255, 159, 64, 0.6)',  // Orange
        'rgba(201, 203, 207, 0.6)', // Gray
        'rgba(255, 99, 132, 0.6)',  // Pink
        'rgba(0, 204, 150, 0.6)'    // Green
    ];

    // Extract provider from model name with special handling for OpenAI models
    function getProvider(modelName) {
        // Convert to lowercase for case-insensitive matching
        const modelLower = modelName.toLowerCase();
        
        // Check for Claude models
        if (modelLower.includes('claude')) {
            return 'claude';
        }
        
        // Check for OpenAI models with various prefixes
        if (modelLower.startsWith('gpt') || 
            modelLower.startsWith('chatgpt') || 
            modelLower.startsWith('o1') || 
            modelLower.startsWith('o3')) {
            return 'openai';
        }
        
        // Check for Gemini models
        if (modelLower.includes('gemini')) {
            return 'gemini';
        }
        
        // Check for Zoo models
        if (modelLower.includes('zoo')) {
            return 'zoo';
        }
        
        // Default: just use the first part before hyphen
        return modelLower.split('-')[0];
    }

    // Create Chart.js datasets with sorting and provider-based coloring
    const createChartDatasets = (metricKey, labelPrefix) => {
        return promptKeys.map((promptKey, index) => {
            // Sort data for this prompt by success rate
            const sortedData = sortModelDataBySuccessRate(promptKey);
            
            return {
                label: `${labelPrefix} (${promptKey})`,
                data: sortedData[metricKey],
                backgroundColor: sortedData.providers.map(provider => {
                    const key = getProvider(provider);
                    return providerColors[key] || defaultColors[index % defaultColors.length];
                }),
                borderColor: sortedData.providers.map(provider => {
                    const key = getProvider(provider);
                    const color = providerColors[key] || defaultColors[index % defaultColors.length];
                    return color.replace(', 0.6', ', 1');
                }),
                borderWidth: 1,
                // Store original model indices for correct label mapping
                originalIndices: sortedData.modelIndices
            };
        });
    };
    // --- Data Preparation for Grouped Bars --- END ---

    // --- >>> DEBUGGING START <<< ---
    console.log("--- Chart Data Debugging ---");
    console.log("Model Names:", modelNames);
    console.log("Prompt Keys:", promptKeys);
    console.log("Populated Datasets:", JSON.parse(JSON.stringify(datasets))); // Deep copy for clean logging
    // --- >>> DEBUGGING END <<< ---

    const commonChartOptions = {
        scales: {
            y: { beginAtZero: true }
        },
        maintainAspectRatio: false,
        plugins: {
             legend: { display: false }, // Hide legend
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


    // --- Chart Creation with Sorted Models ---
    // Helper function to get sorted model labels based on dataset's original indices
    function getSortedLabels(dataset, allModelNames) {
        return dataset.originalIndices.map(index => allModelNames[index]);
    }

    // 1. Overall Success Rate Chart
    const successCtx = document.getElementById('successRateChart').getContext('2d');
    // --- >>> DEBUGGING START <<< ---
    const overallPassRateDatasets = createChartDatasets('overallPassRate', 'Overall Pass Rate');
    console.log("Datasets for Overall Pass Rate Chart:", JSON.parse(JSON.stringify(overallPassRateDatasets)));
    // --- >>> DEBUGGING END <<< ---
    charts.successRate = new Chart(successCtx, {
        type: 'bar',
        data: {
            // Use the sorted labels based on the first dataset
            labels: getSortedLabels(overallPassRateDatasets[0], modelNames),
            datasets: overallPassRateDatasets // Use the logged variable
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Overall Pass Rate (%) by Model & Prompt' }}}
    });

    // 2. Render Success Rate Chart (moved up in order)
    const renderSuccessCtx = document.getElementById('renderSuccessRateChart').getContext('2d');
    const renderSuccessRateDatasets = promptKeys.map(promptKey => 
        createRenderSuccessRateDataset(promptKey, 'Render Success Rate')
    );

    charts.renderSuccessRate = new Chart(renderSuccessCtx, {
        type: 'bar',
        data: {
            labels: getSortedLabels(renderSuccessRateDatasets[0], modelNames),
            datasets: renderSuccessRateDatasets
        },
        options: { 
            ...rateChartOptions, 
            plugins: {
                ...rateChartOptions.plugins, 
                title: { display: true, text: 'STL Render Success Rate (%)' },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const dataset = context.dataset;
                            const index = context.dataIndex;
                            const renderData = dataset._renderData && dataset._renderData[index];
                            let label = dataset.label || '';
                            if (label) label += ': ';
                            
                            if (renderData && typeof renderData === 'object') {
                                const rate = !isNaN(renderData.value) ? renderData.value.toFixed(1) + '%' : 'N/A';
                                const successCount = renderData.successCount || 0;
                                const totalCount = renderData.totalCount || 0;
                                // Use the reliable counts from the data for the tooltip
                                const countInfo = `${successCount}/${totalCount} generated`; 
                                label += `${rate} (${countInfo})`;
                            } else {
                                const value = typeof context.parsed.y !== 'undefined' ? context.parsed.y : NaN;
                                label += !isNaN(value) ? value.toFixed(1) + '%' : 'N/A';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });

    // 3. Success rate if STL rendered Chart
    const checksRunSuccessCtx = document.getElementById('checksRunSuccessRateChart').getContext('2d');
    const checksRunSuccessRateDatasets = promptKeys.map(promptKey => 
        createChecksRunSuccessRateDataset(promptKey, 'Checks Success') // Updated label prefix
    );

    charts.checksRunSuccessRate = new Chart(checksRunSuccessCtx, {
        type: 'bar',
        data: {
            labels: getSortedLabels(checksRunSuccessRateDatasets[0], modelNames),
            datasets: checksRunSuccessRateDatasets
        },
        options: { 
            ...rateChartOptions, 
            plugins: { 
                ...rateChartOptions.plugins, 
                title: { display: true, text: 'Success rate if STL rendered' }, // Updated title
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const dataset = context.dataset;
                            const index = context.dataIndex;
                            const checksRunData = dataset._checksRunData && dataset._checksRunData[index];
                            let label = dataset.label || '';
                            if (label) label += ': ';
                            
                            if (checksRunData && typeof checksRunData === 'object') {
                                const rate = !isNaN(checksRunData.value) ? checksRunData.value.toFixed(1) + '%' : 'N/A';
                                const overallPassCount = checksRunData.overallPassCount || 0;
                                const checksRunCount = checksRunData.checksRunCount || 0;
                                // Updated tooltip text
                                const countInfo = `${overallPassCount}/${checksRunCount} checks passed`; 
                                label += `${rate} (${countInfo})`;
                            } else {
                                const value = typeof context.parsed.y !== 'undefined' ? context.parsed.y : NaN;
                                label += !isNaN(value) ? value.toFixed(1) + '%' : 'N/A';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });

    // 4. Chamfer Pass Rate Chart
    const checksCtx = document.getElementById('checksPassedChart').getContext('2d');
    const chamferPassRateDatasets = createChartDatasets('chamferPassRate', 'Chamfer Pass Rate');
    charts.checksPassed = new Chart(checksCtx, {
        type: 'bar',
        data: {
            labels: getSortedLabels(chamferPassRateDatasets[0], modelNames),
            datasets: chamferPassRateDatasets
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Chamfer Pass Rate (% Rel. to Checks Run)' }}}
    });

    // 5. Average Chamfer Distance Chart
    const chamferCtx = document.getElementById('avgSimilarityChart').getContext('2d');
    const avgChamferDatasets = createChartDatasets('avgChamfer', 'Avg Chamfer (mm)');
    charts.similarity = new Chart(chamferCtx, {
        type: 'bar',
        data: {
            labels: getSortedLabels(avgChamferDatasets[0], modelNames),
            datasets: avgChamferDatasets
        },
        options: { ...distanceChartOptions, plugins: { ...distanceChartOptions.plugins, title: { display: true, text: 'Avg Chamfer Distance (mm, Lower is Better)' }}}
    });

    // 6. Average Hausdorff 95p Distance Chart
    const hausdorffCtx = document.getElementById('avgHausdorffChart').getContext('2d');
    const avgHausdorffDatasets = createChartDatasets('avgHausdorff95p', 'Avg Haus. 95p (mm)');
    charts.avgHausdorff = new Chart(hausdorffCtx, {
        type: 'bar',
        data: {
            labels: getSortedLabels(avgHausdorffDatasets[0], modelNames),
            datasets: avgHausdorffDatasets
        },
        options: { ...distanceChartOptions, plugins: { ...distanceChartOptions.plugins, title: { display: true, text: 'Avg Hausdorff 95p Distance (mm, Lower is Better)' }}}
    });

    // 7. Skipping Volume Diff Chart (Canvas ID: avgVolumeDiffChart)
    // We can hide this canvas or reuse it later if needed.
    const avgVolumeDiffCanvas = document.getElementById('avgVolumeDiffChart');
    if (avgVolumeDiffCanvas && avgVolumeDiffCanvas.parentElement) {
         avgVolumeDiffCanvas.parentElement.style.display = 'none'; // Hide the wrapper div
    }

    // 8. Volume Pass Rate Chart
    const volumePassCtx = document.getElementById('volumePassRateChart').getContext('2d');
    const volumePassRateDatasets = createChartDatasets('volumePassRate', 'Volume Pass Rate');
    charts.volumePassRate = new Chart(volumePassCtx, {
        type: 'bar',
        data: {
            labels: getSortedLabels(volumePassRateDatasets[0], modelNames),
            datasets: volumePassRateDatasets
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Volume Pass Rate (% Rel. to Checks Run)' }}}
    });

    // 9. Hausdorff Pass Rate Chart
    const hausdorffPassCtx = document.getElementById('hausdorffPassRateChart').getContext('2d');
    const hausdorffPassRateDatasets = createChartDatasets('hausdorffPassRate', 'Hausdorff Pass Rate');
    charts.hausdorffPassRate = new Chart(hausdorffPassCtx, {
        type: 'bar',
        data: {
            labels: getSortedLabels(hausdorffPassRateDatasets[0], modelNames),
            datasets: hausdorffPassRateDatasets
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Hausdorff Pass Rate (% Rel. to Checks Run)' }}}
    });

    // --- Task Chart Logic (NEW) --- Start ---
    if (taskStatistics && Object.keys(taskStatistics).length > 0) {
        // Sort tasks for consistent chart order
        const sortedTaskIds = Object.keys(taskStatistics).sort();
        const taskSuccessRates = sortedTaskIds.map(taskId => taskStatistics[taskId].overall_pass_rate);
        
        const taskSuccessCtx = document.getElementById('taskSuccessRateChart').getContext('2d');
        charts.taskSuccessRate = new Chart(taskSuccessCtx, {
            type: 'bar',
            data: {
                labels: sortedTaskIds,
                datasets: [{
                    label: 'Overall Pass Rate',
                    data: taskSuccessRates.map(d => d ?? NaN), // Handle potential null/NaN
                    backgroundColor: 'rgba(75, 192, 192, 0.6)', // Example color (teal)
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: { 
                ...rateChartOptions, // Use rate options (0-100% scale)
                plugins: { 
                    ...rateChartOptions.plugins, 
                    title: { display: true, text: 'Overall Pass Rate (%) by Task (Across Models)' } // Specific title
                }
            }
        });
    } else {
        // Optionally hide the task chart canvas if no data
        const taskChartCanvas = document.getElementById('taskSuccessRateChart');
        if (taskChartCanvas && taskChartCanvas.parentElement) {
            taskChartCanvas.parentElement.style.display = 'none';
        }
    }
    // --- Task Chart Logic (NEW) --- End ---
}

// --- NEW: Render Complexity Chart --- Start ---
function renderComplexityChart(complexityAnalysis) {
    const chartWrapper = document.getElementById('complexityChart').parentElement;
    if (!complexityAnalysis || Object.keys(complexityAnalysis).length === 0) {
        console.warn("No complexityAnalysis data provided for chart.");
        if (chartWrapper) {
            chartWrapper.style.display = 'none'; // Hide the wrapper if no data
        }
        return;
    }
    // Ensure wrapper is visible if we have data
    if (chartWrapper) {
        chartWrapper.style.display = 'block';
    }

    const ctx = document.getElementById('complexityChart').getContext('2d');
    
    // Prepare data
    const opCounts = Object.keys(complexityAnalysis).map(Number).sort((a, b) => a - b);
    const avgPassRates = opCounts.map(count => complexityAnalysis[count].avg_overall_pass_rate);

    // Destroy existing chart if it exists
    charts.complexityChart?.destroy();

    // Use rateChartOptions defined in renderSummaryCharts (assuming it's accessible or redefined)
    // For safety, let's re-establish a basic rate option structure here
    const localRateChartOptions = {
        scales: { 
            y: { beginAtZero: true, max: 100, ticks: { callback: value => value + '%' } },
            x: { // Add title to X-axis
                title: {
                    display: true,
                    text: 'Number of Manual Operations'
                }
            }
        },
        maintainAspectRatio: false,
        plugins: {
             tooltip: { /* Add default tooltip or copy from commonChartOptions if needed */ },
             title: { display: true, text: 'Average Pass Rate (%) by Task Complexity (Manual Operations)' }
        }
    };

    charts.complexityChart = new Chart(ctx, {
        type: 'bar', // Or 'line'
        data: {
            labels: opCounts.map(String), // X-axis labels
            datasets: [{
                label: 'Avg Overall Pass Rate',
                data: avgPassRates,
                backgroundColor: 'rgba(255, 159, 64, 0.6)', // Orange
                borderColor: 'rgba(255, 159, 64, 1)',
                borderWidth: 1
            }]
        },
        options: localRateChartOptions
    });
}
// --- NEW: Render Complexity Chart --- End ---

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
        const modelHeaders = ['Model', 'Prompt', 'Overall Pass (%)', 'SCAD Gen (%)', 'Render (%)', 'Checks Run', 'Chamfer Pass (%)', 'Haus. Pass (%)', 'Vol. Pass (%)', 'Avg Chamfer (mm)', 'Avg Haus. 95p (mm)'];
        modelHeaders.forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            modelHeader.appendChild(th);
        });

        const modelBody = modelTable.createTBody();
        for (const [modelName, promptData] of Object.entries(metaStatistics)) {
            for (const [promptKey, stats] of Object.entries(promptData)) {
                const row = modelBody.insertRow();
                row.insertCell().textContent = modelName;
                row.insertCell().textContent = promptKey;
                row.insertCell().textContent = fmtTable(stats.overall_pass_rate, '%');
                row.insertCell().textContent = fmtTable(stats.scad_generation_success_rate, '%');
                row.insertCell().textContent = fmtTable(stats.render_success_rate, '%');
                row.insertCell().textContent = stats.checks_run_count ?? 'N/A';
                row.insertCell().textContent = fmtTable(stats.chamfer_pass_rate, '%');
                row.insertCell().textContent = fmtTable(stats.hausdorff_pass_rate, '%');
                row.insertCell().textContent = fmtTable(stats.volume_pass_rate, '%');
                row.insertCell().textContent = fmtTable(stats.avg_chamfer, ' mm', 2);
                row.insertCell().textContent = fmtTable(stats.avg_hausdorff_95p, ' mm', 2);
            }
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

    // Define Columns (UPDATED keys to match dashboard_data.json)
    const columns = [
        { key: 'Task ID', header: 'Task ID' },
        { key: 'Rep ID', header: 'Rep ID' },
        { key: 'Prompt', header: 'Prompt' },
        { key: 'Provider', header: 'Provider' },
        { key: 'SCAD Gen', header: 'SCAD Gen', format: formatPassFailText, isBoolean: true }, // Keys match JSON now
        { key: 'Render OK', header: 'Render OK', format: formatPassFailText, isBoolean: true },
        { key: 'Watertight', header: 'Watertight', format: formatPassFailText, isBoolean: true },
        { key: 'Single Comp', header: 'Single Comp', format: formatPassFailText, isBoolean: true },
        { key: 'BBox Acc.', header: 'BBox Acc.', format: formatPassFailText, isBoolean: true, tooltip: "Checks if aligned BBox dims are within tolerance" },
        { key: 'Volume Pass', header: 'Volume Pass', format: formatPassFailText, isBoolean: true, tooltip: "Checks if volume difference % is within threshold" },
        { key: 'Hausdorff Pass', header: 'Hausdorff Pass', format: formatPassFailText, isBoolean: true, tooltip: "Checks if Hausdorff 95p distance is within threshold" },
        { key: 'Chamfer Pass', header: 'Chamfer Pass', format: formatPassFailText, isBoolean: true, tooltip: "Checks if Chamfer distance is within threshold" },
        { key: 'Chamfer (mm)', header: 'Chamfer (mm)' }, // Key matches JSON
        {
            key: 'Hausdorff Dist (95p / 99p mm)', // Key matches JSON
            header: 'Hausdorff Dist (95p / 99p mm)',
            hozAlign: "center",
            headerHozAlign: "center",
        },
        { key: 'Vol Ref (mm³)', header: 'Vol Ref (mm³)' }, // Key matches JSON
        { key: 'Vol Gen (mm³)', header: 'Vol Gen (mm³)' }, // Key matches JSON
        { key: 'BBox Ref (mm)', header: 'BBox Ref (mm)' }, // Key matches JSON
        { key: 'BBox Gen Aligned (mm)', header: 'BBox Gen Aligned (mm)' }, // Key matches JSON
        { key: 'Visualize Cmd', header: 'Visualize Cmd' }, // Special handling below, key doesn't matter as much
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
        // Correct key access for overall status
        const isRowFullySuccessful = result['Overall Passed'] === 'Pass';

        let taskIdCell = null;
        let repIdCell = null;
        let promptCell = null;

        columns.forEach(col => {
            const cell = row.insertCell();
            // Direct access using the potentially-spaced key from dashboard_data.json
            let rawValue = result[col.key];
            let displayValue = rawValue; // Default display value
            let cellStatusClass = '';

            // --- Specific Column Handling ---
            // Visualize Cmd needs original paths, which are NOT in the processed data.
            // We need to decide: Either add those paths back to dashboard_data.json
            // OR remove the visualize button from the dashboard table.
            // For now, let's just make it N/A.
            if (col.key === 'Visualize Cmd') {
                const refPath = result.reference_stl_path; // Get the relative reference path
                const genPath = result.output_stl_path;   // Get the relative generated path
                // Get other details for the title
                const taskId = result['Task ID'];
                const repId = result['Rep ID'];
                const modelName = result['Model Name'];

                if (refPath && genPath && taskId && repId && modelName) {
                    // Construct the title string
                    const title = `Viz: ${taskId}-${modelName}-Rep${repId} (Ref=Green, Gen=Red)`;
                    // Construct the command string with flags and quoted paths
                    displayValue = `python scripts/visualize_comparison.py --ref "${refPath}" --gen "${genPath}" --title "${title}"`;
                    cell.textContent = displayValue;
                    cell.style.fontFamily = 'monospace'; // Use monospace for commands
                    cell.style.cursor = 'pointer';    // Indicate it's clickable
                    cell.title = 'Click to copy command'; // Tooltip
                    // Add click-to-copy functionality
                    cell.addEventListener('click', () => {
                        navigator.clipboard.writeText(displayValue)
                            .then(() => {
                                // Optional: Briefly indicate success (e.g., change text)
                                const originalText = cell.textContent;
                                cell.textContent = 'Copied!';
                                setTimeout(() => { cell.textContent = originalText; }, 1000);
                            })
                            .catch(err => {
                                console.error('Failed to copy command: ', err);
                                alert('Failed to copy command to clipboard.');
                            });
                    });
                } else {
                    // If paths are missing, show N/A
                    displayValue = 'N/A';
                    cell.textContent = displayValue;
                    cell.title = 'Visualize command generation requires STL paths.';
                    cell.style.fontStyle = "italic";
                    cell.style.color = "#888";
                }
            }
            // Handle Hausdorff combined display (key is correct now)
            else if (col.key === 'Hausdorff Dist (95p / 99p mm)') {
                 // rawValue already contains the formatted string like "95p: 1.1340\n99p: 1.3152"
                 if (rawValue === null || typeof rawValue === 'undefined' || rawValue === 'N/A' || rawValue.includes('N/A')) {
                     cell.style.fontStyle = "italic";
                     cell.style.color = "#888";
                     displayValue = 'N/A';
                     cell.innerHTML = displayValue;
                 } else {
                    // Replace newline characters for HTML display
                    displayValue = rawValue.replace(/\\n/g, '<br>');
                    cell.innerHTML = displayValue; // Use innerHTML for <br>
                 }
                 cell.style.whiteSpace = 'normal';
                 cell.style.textAlign = 'center';
            }
            // Handle potentially stringified BBox arrays
            else if (col.key === 'BBox Ref (mm)' || col.key === 'BBox Gen Aligned (mm)') {
                 // The fmt function in process_results might have turned arrays into strings
                 // Let's just display the raw string value from JSON
                 if (rawValue !== null && typeof rawValue !== 'undefined') {
                      displayValue = String(rawValue); // Ensure it's a string
                      cell.textContent = displayValue;
                 } else {
                      cell.textContent = 'N/A';
                 }
            }
            // General formatting for other columns
            else {
                 // Use formatter if defined (applies to Pass/Fail columns)
                 if (col.format) {
                     // Pass the raw value ('Pass', 'Fail', 'N/A', true, false, null)
                     displayValue = col.format(rawValue);
                 } else {
                     // Use raw value, handle null/undefined
                     displayValue = (rawValue !== null && typeof rawValue !== 'undefined') ? rawValue : 'N/A';
                 }
                 cell.textContent = displayValue;
            }

            // --- Determine Cell Status Class ---
            // Base decision on the raw value from JSON before formatting
            if (rawValue === null || typeof rawValue === 'undefined' || rawValue === 'N/A') {
                cellStatusClass = 'status-na';
            } else if (col.isBoolean) { // Check if it's a boolean Pass/Fail column
                // Check against the expected Pass/Fail strings OR boolean true/false
                if (rawValue === 'Pass' || rawValue === true) {
                    cellStatusClass = 'status-yes';
                } else if (rawValue === 'Fail' || rawValue === false) {
                    cellStatusClass = 'status-no';
                } else {
                    cellStatusClass = 'status-na'; // Treat unexpected values as N/A visually
                }
            }
            // Apply the status class
            if (cellStatusClass) {
                cell.classList.add(cellStatusClass);
            }

            // Store specific cells for later row-level styling (using NEW keys)
            if (col.key === 'Task ID') taskIdCell = cell;
            if (col.key === 'Rep ID') repIdCell = cell;
            if (col.key === 'Prompt') promptCell = cell;
        });

        // Style Task ID cell based on overall success (using NEW key)
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
    const summaryContainer = document.getElementById('summary-tables-container');

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

        // Render Charts using meta_statistics, task_statistics, AND results_by_model
        renderSummaryCharts(data.meta_statistics || {}, data.task_statistics || {}, data.results_by_model || {}); // Pass results_by_model

        // --- Render Complexity Chart --- Start ---
        const complexityAnalysis = data.complexity_analysis; // Get the new data section
        renderComplexityChart(complexityAnalysis); // Call the new rendering function
        // --- Render Complexity Chart --- End ---

        // --- Ensure Chart Container is Visible (Removed for Debugging) ---
        // if (chartsContainer) {
        //     chartsContainer.style.display = 'grid'; // Explicitly set display
        //     chartsContainer.style.minHeight = '300px'; // Add minimum height
        //     // Add style rule for canvas elements
        //     const styleSheet = document.createElement("style");
        //     styleSheet.innerText = `#charts-container canvas { min-height: 250px; width: 100% !important; }`;
        //     document.head.appendChild(styleSheet);
        // }
        // --- End Ensure Chart Container is Visible ---

        // Render Grids for each model
        if (Object.keys(data.results_by_model).length > 0) {
            for (const [modelName, modelResults] of Object.entries(data.results_by_model)) {
                // --- Add Model Header BACK --- Start ---
                const modelHeader = document.createElement('h2');
                modelHeader.textContent = `Model: ${modelName}`;
                gridsContainer.appendChild(modelHeader);
                // --- Add Model Header BACK --- End ---

                // Create the table/grid itself (function assumes it appends to gridsContainer indirectly or needs it passed)
                // Assuming createModelHtmlTable takes the container now:
                createModelHtmlTable(modelName, modelResults, gridsContainer); 
            }
        } else {
            gridsContainer.innerHTML = '<p>No model results found in the data.</p>';
        }

        loadingIndicator.style.display = 'none';

        // --- DEBUG LOGGING for Layout --- Start ---
        console.log("--- Layout Debugging --- Style issue ---");
        if (chartsContainer) {
            console.log("Charts Container Element:", chartsContainer);
            const computedStyles = window.getComputedStyle(chartsContainer);
            console.log("Charts Container Computed Styles:", {
                display: computedStyles.display,
                width: computedStyles.width,
                maxWidth: computedStyles.maxWidth,
                marginLeft: computedStyles.marginLeft,
                marginRight: computedStyles.marginRight,
                boxSizing: computedStyles.boxSizing
            });
            console.log("Charts Container offsetWidth:", chartsContainer.offsetWidth);

            const parentElement = chartsContainer.parentElement;
            if (parentElement) {
                console.log("Parent Element:", parentElement);
                 const parentComputedStyles = window.getComputedStyle(parentElement);
                 console.log("Parent Computed Styles:", {
                    display: parentComputedStyles.display,
                    alignItems: parentComputedStyles.alignItems,
                    justifyContent: parentComputedStyles.justifyContent,
                    width: parentComputedStyles.width,
                    boxSizing: parentComputedStyles.boxSizing
                 });
                console.log("Parent offsetWidth:", parentElement.offsetWidth);
            } else {
                console.log("Charts container has no parent element?");
            }
        } else {
            console.log("Could not find #charts-container element for debugging.");
        }
        console.log("--- End Layout Debugging ---");
        // --- DEBUG LOGGING for Layout --- End ---

        // --- Set Final Display Style --- (Moved Here)
        if (chartsContainer) {
            chartsContainer.style.display = 'grid';
        }
        // --- End Set Final Display Style ---

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