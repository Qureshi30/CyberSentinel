(function () {
    const h = React.createElement;
    const MODEL_ORDER = ["xgboost", "random_forest", "svm"];

    const SECTIONS = [
        {
            key: "acc_test_plus",
            title: "Accuracy on KDDTest+ (standard test set)",
            subtitle: "Higher is better",
            maximize: true,
            suffix: "%",
            minScale: 0,
            maxScale: 100,
            badgeType: "best",
        },
        {
            key: "acc_test_21",
            title: "Accuracy on KDDTest-21 (harder - 17 novel attack types)",
            subtitle: "Higher is better",
            maximize: true,
            suffix: "%",
            minScale: 0,
            maxScale: 100,
            badgeType: "best",
        },
        {
            key: "attack_recall_21",
            title: "Attack recall on KDDTest-21 (% of attacks caught)",
            subtitle: "Higher is better",
            maximize: true,
            suffix: "%",
            minScale: 0,
            maxScale: 100,
            badgeType: "best",
        },
        {
            key: "train_val_gap",
            title: "Train / val gap (overfitting check - lower is better)",
            subtitle: "Lower is better",
            maximize: false,
            suffix: "%",
            minScale: 0,
            maxScale: 5,
            badgeType: "fixed",
        },
    ];

    const DIAGNOSTIC_FIELDS = [
        { key: "accuracy", label: "Accuracy", suffix: "%" },
        { key: "precision", label: "Precision", suffix: "%" },
        { key: "recall", label: "Recall", suffix: "%" },
        { key: "specificity", label: "Specificity", suffix: "%" },
        { key: "false_positive_rate", label: "False positive rate", suffix: "%" },
        { key: "false_negative_rate", label: "False negative rate", suffix: "%" },
        { key: "f1_score", label: "F1 score", suffix: "%" },
        { key: "f2_score", label: "F2 score", suffix: "%" },
    ];

    const CONFUSION_FIELDS = [
        { key: "tp", label: "TP" },
        { key: "tn", label: "TN" },
        { key: "fp", label: "FP" },
        { key: "fn", label: "FN" },
    ];

    function safeNumber(value) {
        if (typeof value !== "number" || Number.isNaN(value)) {
            return null;
        }
        return value;
    }

    function formatValue(value, suffix) {
        if (value === null || value === undefined) {
            return "N/A";
        }
        return value.toFixed(1) + suffix;
    }

    function formatCount(value) {
        if (value === null || value === undefined) {
            return "N/A";
        }
        return String(value);
    }

    function getBestModel(models, section) {
        let bestKey = null;
        let bestValue = null;

        MODEL_ORDER.forEach(function (modelKey) {
            const model = models[modelKey];
            if (!model || !model.metrics) {
                return;
            }
            const value = safeNumber(model.metrics[section.key]);
            if (value === null) {
                return;
            }

            if (bestValue === null) {
                bestKey = modelKey;
                bestValue = value;
                return;
            }

            if (section.maximize && value > bestValue) {
                bestKey = modelKey;
                bestValue = value;
            }
            if (!section.maximize && value < bestValue) {
                bestKey = modelKey;
                bestValue = value;
            }
        });

        return bestKey;
    }

    function normalizeWidth(value, section) {
        if (value === null || value === undefined) {
            return 0;
        }

        let ratio = (value - section.minScale) / (section.maxScale - section.minScale);
        if (section.key === "train_val_gap") {
            ratio = 1 - ratio;
        }

        ratio = Math.max(0.03, Math.min(1, ratio));
        return ratio * 100;
    }

    function ModelRow(props) {
        const model = props.model;
        const section = props.section;
        const value = safeNumber(model.metrics[section.key]);
        const isBest = props.bestKey === props.modelKey;
        const shouldShowFixed = section.key === "train_val_gap" && value !== null && value <= 1.0;
        const badgeLabel = isBest ? "best" : (shouldShowFixed ? "fixed" : "");
        const barStyle = {
            width: normalizeWidth(value, section) + "%",
            background: model.color,
        };

        return h("div", { className: "row" }, [
            h("div", { className: "model-name", key: "name" }, model.display_name),
            h("div", { className: "track", key: "track" }, [
                h("div", { className: "bar", style: barStyle, key: "bar" }),
            ]),
            h("div", { className: "value", key: "value" }, formatValue(value, section.suffix)),
            h(
                "div",
                {
                    className: "badge " + (isBest ? "best" : (shouldShowFixed ? "fixed" : "")),
                    key: "badge",
                },
                badgeLabel
            ),
        ]);
    }

    function Section(props) {
        const section = props.section;
        const models = props.models;
        const bestKey = getBestModel(models, section);

        return h("section", { className: "section" }, [
            h("h2", { className: "section-title", key: "title" }, section.title),
            MODEL_ORDER.map(function (modelKey) {
                const model = models[modelKey];
                if (!model) {
                    return null;
                }
                return h(ModelRow, {
                    key: modelKey,
                    modelKey: modelKey,
                    model: model,
                    section: section,
                    bestKey: bestKey,
                });
            }),
        ]);
    }

    function MetricChip(props) {
        const value = props.value;
        const suffix = props.suffix || "";

        return h("div", { className: "metric-chip" }, [
            h("div", { className: "metric-chip-label", key: "label" }, props.label),
            h("div", { className: "metric-chip-value", key: "value" },
                suffix === "%" ? formatValue(value, suffix) : formatCount(value)
            ),
        ]);
    }

    function DiagnosticsSection(props) {
        const models = props.models;

        return h("section", { className: "section diagnostics" }, [
            h("h2", { className: "section-title", key: "title" }, "Confusion matrix and error metrics on KDDTest-21"),
            h("div", { className: "diagnostic-grid", key: "grid" },
                MODEL_ORDER.map(function (modelKey) {
                    const model = models[modelKey];
                    if (!model) {
                        return null;
                    }

                    const confusion = model.metrics.confusion || {};

                    return h("article", { className: "diagnostic-card", key: modelKey }, [
                        h("div", { className: "diagnostic-head", key: "head" }, [
                            h("div", { className: "diagnostic-name", key: "name" }, model.display_name),
                            h("div", { className: "diagnostic-accuracy", key: "acc" }, formatValue(model.metrics.acc_test_21, "%")),
                        ]),
                        h("div", { className: "confusion-grid", key: "confusion" },
                            CONFUSION_FIELDS.map(function (field) {
                                return h("div", { className: "confusion-cell", key: field.key }, [
                                    h("div", { className: "confusion-label", key: "l" }, field.label),
                                    h("div", { className: "confusion-value", key: "v" }, formatCount(confusion[field.key])),
                                ]);
                            })
                        ),
                        h("div", { className: "chip-grid", key: "chips" },
                            DIAGNOSTIC_FIELDS.map(function (field) {
                                return h(MetricChip, {
                                    key: field.key,
                                    label: field.label,
                                    value: confusion[field.key],
                                    suffix: field.suffix,
                                });
                            })
                        ),
                    ]);
                })
            ),
        ]);
    }

    function App() {
        const state = React.useState({ loading: true, error: null, data: null });
        const view = state[0];
        const setView = state[1];
        const predictState = React.useState({
            running: false,
            error: null,
            payload: null,
            algorithm: "xgboost",
            pipeline: "cascade",
        });
        const predictView = predictState[0];
        const setPredictView = predictState[1];

        React.useEffect(function () {
            fetch("/api/dashboard-metrics")
                .then(function (res) {
                    if (!res.ok) {
                        throw new Error("Failed to load metrics");
                    }
                    return res.json();
                })
                .then(function (json) {
                    setView({ loading: false, error: null, data: json });
                })
                .catch(function (err) {
                    setView({ loading: false, error: err.message, data: null });
                });
        }, []);

        if (view.loading) {
            return h("div", { className: "shell" }, [
                h("h1", { className: "headline", key: "h" }, "CyberSentinel Model Dashboard"),
                h("div", { className: "loading", key: "l" }, "Loading live model metrics..."),
            ]);
        }

        if (view.error) {
            return h("div", { className: "shell" }, [
                h("h1", { className: "headline", key: "h" }, "CyberSentinel Model Dashboard"),
                h("div", { className: "error", key: "e" }, "Unable to load metrics: " + view.error),
            ]);
        }

        const generatedAt = view.data.generated_at ? new Date(view.data.generated_at).toLocaleString() : "Unknown";

        function onUploadSubmit(ev) {
            ev.preventDefault();
            const formEl = ev.currentTarget;
            const fileInput = formEl.querySelector("input[name='file']");
            const algorithmSelect = formEl.querySelector("select[name='algorithm']");
            const pipelineSelect = formEl.querySelector("select[name='pipeline']");

            const file = fileInput && fileInput.files ? fileInput.files[0] : null;
            const algorithm = algorithmSelect ? algorithmSelect.value : "xgboost";
            const pipeline = pipelineSelect ? pipelineSelect.value : "cascade";

            if (!file) {
                setPredictView({
                    running: false,
                    error: "Please select a CSV file before running prediction.",
                    payload: null,
                    algorithm: algorithm,
                    pipeline: pipeline,
                });
                return;
            }

            const fd = new FormData();
            fd.append("file", file);
            fd.append("algorithm", algorithm);
            fd.append("pipeline", pipeline);

            setPredictView({
                running: true,
                error: null,
                payload: null,
                algorithm: algorithm,
                pipeline: pipeline,
            });

            fetch("/api/predict", {
                method: "POST",
                body: fd,
            })
                .then(function (res) {
                    return res.json().then(function (json) {
                        if (!res.ok) {
                            throw new Error(json.error || "Prediction failed");
                        }
                        return json;
                    });
                })
                .then(function (json) {
                    setPredictView({
                        running: false,
                        error: null,
                        payload: json,
                        algorithm: algorithm,
                        pipeline: pipeline,
                    });
                })
                .catch(function (err) {
                    setPredictView({
                        running: false,
                        error: err.message,
                        payload: null,
                        algorithm: algorithm,
                        pipeline: pipeline,
                    });
                });
        }

        var predictionPanel = null;
        if (predictView.payload) {
            const rows = predictView.payload.rows || [];
            const summary = predictView.payload.summary || {};
            predictionPanel = h("section", { className: "section inference-results" }, [
                h("h2", { className: "section-title", key: "title" }, "Live Prediction Results"),
                h("p", { className: "subline", key: "meta" },
                    "Pipeline: " + (predictView.payload.pipeline || predictView.pipeline) +
                    " | Model: " + (predictView.payload.algorithm || predictView.algorithm)
                ),
                h("div", { className: "result-summary", key: "summary" }, [
                    h("div", { className: "result-pill", key: "t" }, "Total: " + (summary.total || 0)),
                    h("div", { className: "result-pill", key: "n" }, "Normal: " + (summary.normal || 0)),
                    h("div", { className: "result-pill", key: "a" }, "Attack: " + (summary.attack || 0)),
                    h("div", { className: "result-pill", key: "s" }, "Showing: " + (summary.showing || 0)),
                ]),
                h("div", { className: "result-table-wrap", key: "table" }, [
                    h("table", { className: "result-table" }, [
                        h("thead", { key: "thead" }, [
                            h("tr", null, [
                                h("th", { key: "r" }, "#"),
                                h("th", { key: "p" }, "Prediction"),
                                h("th", { key: "f" }, "Details"),
                            ]),
                        ]),
                        h("tbody", { key: "tbody" },
                            rows.map(function (r) {
                                var details = Object.keys(r.fields || {}).map(function (k) {
                                    return k + ": " + String(r.fields[k]);
                                }).join(" | ");

                                return h("tr", { key: "row-" + r.row_number }, [
                                    h("td", { key: "c1" }, String(r.row_number)),
                                    h("td", { key: "c2" }, String(r.prediction)),
                                    h("td", { key: "c3", className: "details-cell" }, details),
                                ]);
                            })
                        ),
                    ]),
                ]),
            ]);
        }

        return h("main", { className: "shell" }, [
            h("h1", { className: "headline", key: "headline" }, "CyberSentinel Model Dashboard"),
            h(
                "p",
                { className: "subline", key: "subline" },
                "This compares all three models on the key metrics using your latest generated prediction files."
            ),
            h("section", { className: "section upload-section", key: "upload" }, [
                h("h2", { className: "section-title", key: "upload-title" }, "Run Live Prediction"),
                h("form", { className: "upload-form", onSubmit: onUploadSubmit, key: "upload-form" }, [
                    h("input", {
                        type: "file",
                        name: "file",
                        accept: ".csv",
                        required: true,
                        key: "file",
                    }),
                    h("select", { name: "pipeline", defaultValue: predictView.pipeline, key: "pipeline" }, [
                        h("option", { value: "cascade", key: "c" }, "cascade"),
                        h("option", { value: "single", key: "s" }, "single"),
                    ]),
                    h("select", { name: "algorithm", defaultValue: predictView.algorithm, key: "algorithm" }, [
                        h("option", { value: "xgboost", key: "x" }, "xgboost"),
                        h("option", { value: "randomforest", key: "r" }, "randomforest"),
                        h("option", { value: "svm", key: "v" }, "svm"),
                    ]),
                    h("button", { type: "submit", disabled: predictView.running, key: "run" },
                        predictView.running ? "Running..." : "Predict"
                    ),
                ]),
                predictView.error
                    ? h("div", { className: "error", key: "upload-error" }, predictView.error)
                    : null,
            ]),
            predictionPanel,
            SECTIONS.map(function (section) {
                return h(Section, {
                    key: section.key,
                    section: section,
                    models: view.data.models || {},
                });
            }),
            h(DiagnosticsSection, {
                key: "diagnostics",
                models: view.data.models || {},
            }),
            h("div", { className: "footer-note", key: "footer" }, "Last updated: " + generatedAt),
        ]);
    }

    const root = ReactDOM.createRoot(document.getElementById("app"));
    root.render(h(App));
})();
