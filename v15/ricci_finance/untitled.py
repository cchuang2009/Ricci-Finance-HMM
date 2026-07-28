def round_numeric(value, digits: int = ROUND_DIGITS):
    """Round scalar float-like values while preserving non-numeric objects."""
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def round_dataframe(df: pd.DataFrame, digits: int = ROUND_DIGITS) -> pd.DataFrame:
    """Return a copy with all numeric columns rounded for display/export."""
    result = df.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(digits)
    return result


def round_graph_inplace(graph: nx.Graph, digits: int = ROUND_DIGITS) -> nx.Graph:
    """Round every floating-point node and edge attribute in a graph."""
    for _, attributes in graph.nodes(data=True):
        for key, value in list(attributes.items()):
            attributes[key] = round_numeric(value, digits)
    for _, _, attributes in graph.edges(data=True):
        for key, value in list(attributes.items()):
            attributes[key] = round_numeric(value, digits)
    return graph


def round_nested(value, digits: int = ROUND_DIGITS):
    """Recursively round values used in ECharts options and JSON output."""
    if isinstance(value, dict):
        return {key: round_nested(item, digits) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(round_nested(item, digits) for item in value)
    if isinstance(value, list):
        return [round_nested(item, digits) for item in value]
    if isinstance(value, np.ndarray):
        return [round_nested(item, digits) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    return round_numeric(value, digits)


def round_plotly_figure(figure, digits: int = ROUND_DIGITS):
    """Rebuild a Plotly figure with all numeric JSON values rounded."""
    return go.Figure(round_nested(figure.to_dict(), digits))


def capital_flow_animation_options(
    aligned_frames: list[dict],
    close: pd.DataFrame,
    sectors: dict[str, str],
    interval_ms: int = 900,
) -> dict:
    """Build an animated sector-flow matrix with gradual change colors.

    Each cell label is the current dimensionless flow score. Its background
    color represents the cell's change from the preceding frame: increasingly
    green for an increase, increasingly red for a decrease, and near-neutral
    for little or no change. The score is model-derived and is not USD.
    """
    snapshots: list[tuple[str, pd.DataFrame]] = []
    all_sector_names: set[str] = set()

    for frame in aligned_frames:
        graph = round_graph_inplace(frame["graph"].copy())
        date = pd.Timestamp(frame["date"])
        momentum = sector_momentum(close, sectors, date).round(ROUND_DIGITS)
        flow = sector_flow_matrix(graph, sectors, momentum).round(ROUND_DIGITS)
        if flow.empty:
            continue
        flow.index = flow.index.astype(str)
        flow.columns = flow.columns.astype(str)
        all_sector_names.update(flow.index)
        all_sector_names.update(flow.columns)
        snapshots.append((str(date.date()), flow))

    if not snapshots:
        return {
            "title": {"text": "No capital-flow animation data"},
            "series": [],
        }

    sector_names = sorted(all_sector_names)
    normalized: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    change_values: list[float] = []
    previous_matrix: pd.DataFrame | None = None

    for date_label, flow in snapshots:
        matrix = flow.reindex(
            index=sector_names, columns=sector_names, fill_value=0.0
        ).fillna(0.0).round(ROUND_DIGITS)

        change = (
            pd.DataFrame(0.0, index=sector_names, columns=sector_names)
            if previous_matrix is None
            else matrix - previous_matrix
        ).round(ROUND_DIGITS)

        change_values.extend(change.to_numpy(dtype=float).ravel().tolist())
        normalized.append((date_label, matrix, change))
        previous_matrix = matrix.copy()

    change_abs_max = max((abs(float(v)) for v in change_values), default=1.0)
    change_abs_max = max(round(change_abs_max, ROUND_DIGITS), 0.001)

    options = []
    for date_label, matrix, change in normalized:
        # [destination index, source index, current score, change from prior frame]
        heatmap_data = [
            [
                column_index,
                row_index,
                round(float(matrix.iloc[row_index, column_index]), ROUND_DIGITS),
                round(float(change.iloc[row_index, column_index]), ROUND_DIGITS),
            ]
            for row_index in range(len(sector_names))
            for column_index in range(len(sector_names))
        ]

        options.append({
            "title": {
                "text": f"Sector capital-flow change — {date_label}",
                "subtext": (
                    "Cell number = current dimensionless flow score; "
                    "cell color = change from previous frame"
                ),
                "left": "center",
                "textStyle": {"fontSize": 24, "fontWeight": "bold"},
                "subtextStyle": {"fontSize": 16},
            },
            "series": [{
                "name": "Capital-flow score and change",
                "type": "heatmap",
                "data": heatmap_data,
                "label": {
                    "show": True,
                    "formatter": "{@[2]}",
                    "fontSize": 16,
                    "fontWeight": "bold",
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowColor": "rgba(0,0,0,0.35)",
                    }
                },
            }],
        })

    return round_nested({
        "baseOption": {
            "timeline": {
                "axisType": "category",
                "autoPlay": True,
                "playInterval": int(interval_ms),
                "loop": True,
                "bottom": 0,
                "data": [date_label for date_label, *_ in normalized],
                "label": {"formatter": "{value}", "fontSize": 14},
                "controlStyle": {"itemSize": 22},
            },
            "tooltip": {
                "trigger": "item",
                "textStyle": {"fontSize": 15},
                "formatter": (
                    "function (p) {"
                    "var current = Number(p.value[2]).toFixed(3);"
                    "var delta = Number(p.value[3]).toFixed(3);"
                    "var sign = Number(p.value[3]) > 0 ? '+' : '';"
                    "return '<b>' + p.name + '</b><br/>' +"
                    "'Current score: ' + current + '<br/>' +"
                    "'Change: ' + sign + delta;"
                    "}"
                ),
            },
            "grid": {"top": 110, "left": 150, "right": 125, "bottom": 105},
            "xAxis": {
                "type": "category",
                "data": sector_names,
                "name": "Destination sector",
                "nameTextStyle": {"fontSize": 18, "fontWeight": "bold"},
                "splitArea": {"show": True},
                "axisLabel": {"rotate": 25, "fontSize": 17, "fontWeight": "bold"},
            },
            "yAxis": {
                "type": "category",
                "data": sector_names,
                "name": "Source sector",
                "nameTextStyle": {"fontSize": 18, "fontWeight": "bold"},
                "splitArea": {"show": True},
                "axisLabel": {"fontSize": 17, "fontWeight": "bold"},
            },
            "visualMap": {
                "type": "continuous",
                "dimension": 3,
                "min": -change_abs_max,
                "max": change_abs_max,
                "calculable": True,
                "orient": "vertical",
                "right": 5,
                "top": 145,
                "precision": ROUND_DIGITS,
                "text": ["Increase", "Decrease"],
                "textStyle": {"fontSize": 15, "fontWeight": "bold"},
                "inRange": {
                    "color": [
                        "#8b1d1d",
                        "#d96b6b",
                        "#f4d4d4",
                        "#f2f2f2",
                        "#d3ecd9",
                        "#65b87a",
                        "#176b35",
                    ]
                },
            },
            "series": [{"type": "heatmap", "data": []}],
            "animationDurationUpdate": min(int(interval_ms * 0.75), 700),
            "animationEasingUpdate": "cubicInOut",
        },
        "options": options,
    })
