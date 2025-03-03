# src/reports/visualization.py
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import json
from datetime import datetime, timedelta
from loguru import logger
import asyncio
import os
import numpy as np
import pandas as pd
import json
import aiofiles
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from loguru import logger

from src.models.entities import Transaction, WalletAnalysis, Wave
from config.settings import TransactionType


class ChartGenerator:
    """
    Generates chart configurations for various data visualizations.
    These configurations can be used with charting libraries like Chart.js.
    """

    @staticmethod
    def generate_time_series_config(
            data: pd.DataFrame,
            date_column: str,
            value_column: str,
            title: str = "Time Series",
            x_axis_label: str = "Date",
            y_axis_label: str = "Value"
    ) -> Dict[str, Any]:
        """
        Generate a time series chart configuration.

        Args:
            data: DataFrame containing time series data
            date_column: Name of the date column
            value_column: Name of the value column
            title: Chart title
            x_axis_label: X-axis label
            y_axis_label: Y-axis label

        Returns:
            Chart.js compatible configuration
        """
        if data.empty:
            return {"error": "No data available for time series chart"}

        # Sort data by date
        data = data.sort_values(by=date_column)

        # Format dates and values
        dates = data[date_column].astype(str).tolist()
        values = data[value_column].tolist()

        return {
            "type": "line",
            "data": {
                "labels": dates,
                "datasets": [
                    {
                        "label": y_axis_label,
                        "data": values,
                        "borderColor": "rgba(75, 192, 192, 1)",
                        "backgroundColor": "rgba(75, 192, 192, 0.2)",
                        "borderWidth": 2,
                        "tension": 0.1,
                        "fill": True
                    }
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": title
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": x_axis_label
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": y_axis_label
                        },
                        "beginAtZero": True
                    }
                }
            }
        }

    @staticmethod
    def generate_bar_chart_config(
            data: pd.DataFrame,
            category_column: str,
            value_column: str,
            title: str = "Bar Chart",
            x_axis_label: str = "Categories",
            y_axis_label: str = "Value"
    ) -> Dict[str, Any]:
        """
        Generate a bar chart configuration.

        Args:
            data: DataFrame containing bar chart data
            category_column: Name of the category column
            value_column: Name of the value column
            title: Chart title
            x_axis_label: X-axis label
            y_axis_label: Y-axis label

        Returns:
            Chart.js compatible configuration
        """
        if data.empty:
            return {"error": "No data available for bar chart"}

        # Sort data by value (descending)
        data = data.sort_values(by=value_column, ascending=False)

        # Format categories and values
        categories = data[category_column].astype(str).tolist()
        values = data[value_column].tolist()

        return {
            "type": "bar",
            "data": {
                "labels": categories,
                "datasets": [
                    {
                        "label": y_axis_label,
                        "data": values,
                        "backgroundColor": "rgba(54, 162, 235, 0.5)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "borderWidth": 1
                    }
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": title
                    },
                    "legend": {
                        "display": False
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": x_axis_label
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": y_axis_label
                        },
                        "beginAtZero": True
                    }
                }
            }
        }

    @staticmethod
    def generate_pie_chart_config(
            data: pd.DataFrame,
            category_column: str,
            value_column: str,
            title: str = "Pie Chart"
    ) -> Dict[str, Any]:
        """
        Generate a pie chart configuration.

        Args:
            data: DataFrame containing pie chart data
            category_column: Name of the category column
            value_column: Name of the value column
            title: Chart title

        Returns:
            Chart.js compatible configuration
        """
        if data.empty:
            return {"error": "No data available for pie chart"}

        # Sort data by value (descending)
        data = data.sort_values(by=value_column, ascending=False)

        # Format categories and values
        categories = data[category_column].astype(str).tolist()
        values = data[value_column].tolist()

        # Generate colors
        colors = [
            "rgba(255, 99, 132, 0.7)",
            "rgba(54, 162, 235, 0.7)",
            "rgba(255, 206, 86, 0.7)",
            "rgba(75, 192, 192, 0.7)",
            "rgba(153, 102, 255, 0.7)",
            "rgba(255, 159, 64, 0.7)",
            "rgba(199, 199, 199, 0.7)",
            "rgba(83, 102, 255, 0.7)",
            "rgba(40, 159, 64, 0.7)",
            "rgba(210, 199, 199, 0.7)",
        ]

        # Repeat colors if there are more categories than colors
        if len(categories) > len(colors):
            colors = colors * (len(categories) // len(colors) + 1)

        return {
            "type": "pie",
            "data": {
                "labels": categories,
                "datasets": [
                    {
                        "data": values,
                        "backgroundColor": colors[:len(categories)],
                        "borderWidth": 1
                    }
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": title
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False
                    }
                }
            }
        }

    @staticmethod
    def generate_heatmap_config(
            data: pd.DataFrame,
            x_column: str,
            y_column: str,
            value_column: str,
            title: str = "Heatmap",
            x_axis_label: str = "X-Axis",
            y_axis_label: str = "Y-Axis"
    ) -> Dict[str, Any]:
        """
        Generate a heatmap configuration.

        Args:
            data: DataFrame containing heatmap data
            x_column: Name of the x-axis column
            y_column: Name of the y-axis column
            value_column: Name of the value column
            title: Chart title
            x_axis_label: X-axis label
            y_axis_label: Y-axis label

        Returns:
            Chart.js compatible configuration
        """
        if data.empty:
            return {"error": "No data available for heatmap"}

        # Get unique x and y values
        x_values = sorted(data[x_column].unique().tolist())
        y_values = sorted(data[y_column].unique().tolist())

        # Create a 2D grid for heatmap data
        grid_data = []
        for y in y_values:
            row_data = []
            for x in x_values:
                filtered = data[(data[x_column] == x) & (data[y_column] == y)]
                if not filtered.empty:
                    row_data.append(filtered[value_column].values[0])
                else:
                    row_data.append(0)  # Default value for missing data points
            grid_data.append(row_data)

        return {
            "type": "matrix",  # Note: This is a custom type for heatmap
            "data": {
                "x_labels": x_values,
                "y_labels": y_values,
                "values": grid_data
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": title
                    },
                    "tooltip": {
                        "mode": "nearest",
                        "intersect": False
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": x_axis_label
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": y_axis_label
                        }
                    }
                }
            }
        }

    @staticmethod
    def generate_comparison_chart_config(
            data: pd.DataFrame,
            date_column: str,
            series_columns: List[str],
            title: str = "Comparison Chart",
            x_axis_label: str = "Date",
            y_axis_label: str = "Value"
    ) -> Dict[str, Any]:
        """
        Generate a comparison chart configuration.

        Args:
            data: DataFrame containing time series data
            date_column: Name of the date column
            series_columns: Names of the columns to compare
            title: Chart title
            x_axis_label: X-axis label
            y_axis_label: Y-axis label

        Returns:
            Chart.js compatible configuration
        """
        if data.empty or not all(col in data.columns for col in series_columns):
            return {"error": "No data available for comparison chart or missing columns"}

        # Sort data by date
        data = data.sort_values(by=date_column)

        # Format dates
        dates = data[date_column].astype(str).tolist()

        # Generate colors
        colors = [
            "rgba(255, 99, 132, 1)",
            "rgba(54, 162, 235, 1)",
            "rgba(255, 206, 86, 1)",
            "rgba(75, 192, 192, 1)",
            "rgba(153, 102, 255, 1)",
            "rgba(255, 159, 64, 1)",
            "rgba(199, 199, 199, 1)",
            "rgba(83, 102, 255, 1)",
            "rgba(40, 159, 64, 1)",
            "rgba(210, 199, 199, 1)",
        ]

        # Prepare datasets
        datasets = []
        for i, column in enumerate(series_columns):
            color_index = i % len(colors)
            datasets.append({
                "label": column,
                "data": data[column].tolist(),
                "borderColor": colors[color_index],
                "backgroundColor": colors[color_index].replace("1)", "0.2)"),
                "borderWidth": 2,
                "fill": False
            })

        return {
            "type": "line",
            "data": {
                "labels": dates,
                "datasets": datasets
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": title
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": x_axis_label
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": y_axis_label
                        }
                    }
                }
            }
        }

    @staticmethod
    def generate_timeline_config(
            data: pd.DataFrame,
            start_date_column: str,
            end_date_column: str,
            category_column: str,
            value_column: str = None,
            title: str = "Timeline Chart"
    ) -> Dict[str, Any]:
        """
        Generate a timeline chart configuration.

        Args:
            data: DataFrame containing timeline data
            start_date_column: Name of the start date column
            end_date_column: Name of the end date column
            category_column: Name of the category column (for grouping)
            value_column: Optional name of the value column (for sizing)
            title: Chart title

        Returns:
            Chart.js compatible configuration for timeline visualization
        """
        if data.empty:
            return {"error": "No data available for timeline chart"}

        # Group by category
        categories = sorted(data[category_column].unique().tolist())

        # Prepare timeline events
        events = []
        for _, row in data.iterrows():
            event = {
                "group": row[category_column],
                "start": row[start_date_column],
                "end": row[end_date_column],
                "content": f"{row[category_column]} - {row[value_column] if value_column else ''}"
            }

            # Add value if available
            if value_column and value_column in row:
                event["value"] = row[value_column]

            events.append(event)

        return {
            "type": "timeline",  # Custom type for timeline visualization
            "data": {
                "groups": categories,
                "events": events
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": title
                    }
                }
            }
        }

    @staticmethod
    def wallet_behavior_chart(
            wallet_data: pd.DataFrame,
            title: str = "Wallet Behavior Distribution"
    ) -> Dict[str, Any]:
        """
        Generate a chart configuration for wallet behavior analysis.

        Args:
            wallet_data: DataFrame containing wallet classification data
            title: Chart title

        Returns:
            Chart.js compatible configuration
        """
        if wallet_data.empty or "classification" not in wallet_data.columns:
            return {"error": "No classification data available for wallet behavior chart"}

        # Count wallets by classification
        counts = wallet_data["classification"].value_counts().reset_index()
        counts.columns = ["classification", "count"]

        # Generate pie chart for wallet classifications
        return ChartGenerator.generate_pie_chart_config(
            data=counts,
            category_column="classification",
            value_column="count",
            title=title
        )

    @staticmethod
    def generate_wave_timeline(
            buy_waves: pd.DataFrame,
            sell_waves: pd.DataFrame,
            start_timestamp_col: str = "start_timestamp",
            end_timestamp_col: str = "end_timestamp",
            amount_col: str = "total_amount",
            title: str = "Buy/Sell Wave Timeline"
    ) -> Dict[str, Any]:
        """
        Generate a timeline visualization for buy and sell waves.

        Args:
            buy_waves: DataFrame containing buy wave data
            sell_waves: DataFrame containing sell wave data
            start_timestamp_col: Name of the start timestamp column
            end_timestamp_col: Name of the end timestamp column
            amount_col: Name of the amount column
            title: Chart title

        Returns:
            Chart.js compatible configuration for wave timeline
        """
        # Add wave type column
        if not buy_waves.empty:
            buy_waves = buy_waves.copy()
            buy_waves["wave_type"] = "BUY"

        if not sell_waves.empty:
            sell_waves = sell_waves.copy()
            sell_waves["wave_type"] = "SELL"

        # Combine waves
        all_waves = pd.concat([buy_waves, sell_waves])

        if all_waves.empty:
            return {"error": "No wave data available for timeline"}

        # Generate timeline configuration
        return ChartGenerator.generate_timeline_config(
            data=all_waves,
            start_date_column=start_timestamp_col,
            end_date_column=end_timestamp_col,
            category_column="wave_type",
            value_column=amount_col,
            title=title
        )

    @staticmethod
    def generate_pattern_distribution(
            patterns: pd.DataFrame,
            type_column: str = "type",
            strength_column: str = "strength",
            title: str = "Pattern Distribution"
    ) -> Dict[str, Any]:
        """
        Generate a chart configuration for pattern distribution.

        Args:
            patterns: DataFrame containing pattern data
            type_column: Name of the pattern type column
            strength_column: Name of the pattern strength column
            title: Chart title

        Returns:
            Chart.js compatible configuration
        """
        if patterns.empty or type_column not in patterns.columns:
            return {"error": "No pattern data available for distribution chart"}

        # Count patterns by type
        counts = patterns[type_column].value_counts().reset_index()
        counts.columns = ["pattern_type", "count"]

        # Generate pie chart for pattern distribution
        return ChartGenerator.generate_pie_chart_config(
            data=counts,
            category_column="pattern_type",
            value_column="count",
            title=title
        )

    @staticmethod
    def generate_transaction_volume_chart(
            transactions: pd.DataFrame,
            timestamp_column: str = "timestamp",
            amount_column: str = "amount",
            transaction_type_column: str = "transaction_type",
            group_by: str = "day",
            title: str = "Transaction Volume Over Time"
    ) -> Dict[str, Any]:
        """
        Generate a chart configuration for transaction volume over time.

        Args:
            transactions: DataFrame containing transaction data
            timestamp_column: Name of the timestamp column
            amount_column: Name of the amount column
            transaction_type_column: Name of the transaction type column
            group_by: Grouping period ('hour', 'day', 'week', 'month')
            title: Chart title

        Returns:
            Chart.js compatible configuration
        """
        if transactions.empty:
            return {"error": "No transaction data available for volume chart"}

        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_dtype(transactions[timestamp_column]):
            transactions = transactions.copy()
            transactions[timestamp_column] = pd.to_datetime(transactions[timestamp_column])

        # Create date column based on grouping
        if group_by == "hour":
            transactions["date"] = transactions[timestamp_column].dt.strftime("%Y-%m-%d %H:00")
        elif group_by == "day":
            transactions["date"] = transactions[timestamp_column].dt.strftime("%Y-%m-%d")
        elif group_by == "week":
            transactions["date"] = transactions[timestamp_column].dt.to_period("W").dt.start_time.dt.strftime(
                "%Y-%m-%d")
        elif group_by == "month":
            transactions["date"] = transactions[timestamp_column].dt.strftime("%Y-%m")
        else:
            transactions["date"] = transactions[timestamp_column].dt.strftime("%Y-%m-%d")

        # Group by date and transaction type
        if transaction_type_column in transactions.columns:
            # Separate buy and sell transactions
            buy_volume = transactions[transactions[transaction_type_column] == "BUY"].groupby("date")[
                amount_column].sum().reset_index()
            buy_volume.columns = ["date", "buy_volume"]

            sell_volume = transactions[transactions[transaction_type_column] == "SELL"].groupby("date")[
                amount_column].sum().reset_index()
            sell_volume.columns = ["date", "sell_volume"]

            # Merge buy and sell data
            volume_data = pd.merge(buy_volume, sell_volume, on="date", how="outer").fillna(0)

            # Generate comparison chart
            return ChartGenerator.generate_comparison_chart_config(
                data=volume_data,
                date_column="date",
                series_columns=["buy_volume", "sell_volume"],
                title=title,
                x_axis_label="Date",
                y_axis_label="Volume"
            )
        else:
            # Group by date only
            volume_data = transactions.groupby("date")[amount_column].sum().reset_index()

            # Generate time series chart
            return ChartGenerator.generate_time_series_config(
                data=volume_data,
                date_column="date",
                value_column=amount_column,
                title=title,
                x_axis_label="Date",
                y_axis_label="Volume"
            )


class VisualizationRenderer:
    """
    Renders chart configurations into HTML, SVG, or other formats.
    """

    @staticmethod
    def chart_config_to_html(chart_config: Dict[str, Any], element_id: str, width: str = "100%",
                             height: str = "400px") -> str:
        """
        Convert chart configuration to HTML with Chart.js.

        Args:
            chart_config: Chart.js compatible configuration
            element_id: HTML element ID for the chart
            width: Width of the chart
            height: Height of the chart

        Returns:
            HTML string with Chart.js initialization
        """
        # Special handling for timeline charts
        if chart_config.get("type") == "timeline":
            return VisualizationRenderer._timeline_to_html(chart_config, element_id, width, height)

        # Special handling for matrix/heatmap charts
        if chart_config.get("type") == "matrix":
            return VisualizationRenderer._heatmap_to_html(chart_config, element_id, width, height)

        # Regular Chart.js charts
        return f"""
        <div style="width: {width}; height: {height};">
            <canvas id="{element_id}"></canvas>
        </div>
        <script>
            (function() {{
                const ctx = document.getElementById('{element_id}').getContext('2d');
                new Chart(ctx, {json.dumps(chart_config)});
            }})();
        </script>
        """

    @staticmethod
    def _timeline_to_html(chart_config: Dict[str, Any], element_id: str, width: str = "100%",
                          height: str = "400px") -> str:
        """
        Convert timeline configuration to HTML with vis-timeline.

        Args:
            chart_config: Timeline configuration
            element_id: HTML element ID for the timeline
            width: Width of the timeline
            height: Height of the timeline

        Returns:
            HTML string with vis-timeline initialization
        """
        # Extract data from chart config
        timeline_data = chart_config.get("data", {})
        groups = timeline_data.get("groups", [])
        events = timeline_data.get("events", [])
        title = chart_config.get("options", {}).get("plugins", {}).get("title", {}).get("text", "Timeline")

        # Create group items JSON
        group_items = []
        for i, group in enumerate(groups):
            group_items.append({
                "id": i,
                "content": group
            })

        # Create event items JSON
        event_items = []
        for i, event in enumerate(events):
            group_index = groups.index(event["group"]) if event["group"] in groups else 0
            event_items.append({
                "id": i + 100,  # Avoid potential ID conflicts with groups
                "group": group_index,
                "start": event["start"],
                "end": event["end"],
                "content": event["content"],
                "title": event["content"]
            })

        return f"""
        <div style="width: {width}; height: {height};">
            <h3>{title}</h3>
            <div id="{element_id}"></div>
        </div>
        <script>
            (function() {{
                // Create groups
                const groups = new vis.DataSet({json.dumps(group_items)});

                // Create items
                const items = new vis.DataSet({json.dumps(event_items)});

                // Create timeline
                const container = document.getElementById('{element_id}');
                const options = {{
                    stack: true,
                    horizontalScroll: true,
                    zoomKey: 'ctrlKey',
                    maxHeight: '{height}',
                    min: new Date({json.dumps(min(events, key=lambda e: e['start'])['start']).replace('"', '')}),
                    max: new Date({json.dumps(max(events, key=lambda e: e['end'])['end']).replace('"', '')})
                }};

                new vis.Timeline(container, items, groups, options);
            }})();
        </script>
        """

        # Continue implementation of src/reports/visualization.py

        @staticmethod
        def _heatmap_to_html(chart_config: Dict[str, Any], element_id: str, width: str = "100%",
                             height: str = "400px") -> str:

            """
            Convert heatmap configuration to HTML with D3.js.

            Args:
                chart_config: Heatmap configuration
                element_id: HTML element ID for the heatmap
                width: Width of the heatmap
                height: Height of the heatmap

            Returns:
                HTML string with D3.js heatmap initialization
            """
            # Extract data from chart config
            matrix_data = chart_config.get("data", {})
            x_labels = matrix_data.get("x_labels", [])
            y_labels = matrix_data.get("y_labels", [])
            values = matrix_data.get("values", [])
            title = chart_config.get("options", {}).get("plugins", {}).get("title", {}).get("text", "Heatmap")

            return f"""
            <div style="width: {width}; height: {height};">
                <h3>{title}</h3>
                <div id="{element_id}"></div>
            </div>
            <script>
                (function() {{
                    // Data
                    const xLabels = {json.dumps(x_labels)};
                    const yLabels = {json.dumps(y_labels)};
                    const data = {json.dumps(values)};

                    // Dimensions
                    const margin = {{top: 50, right: 70, bottom: 100, left: 100}};
                    const width = document.getElementById('{element_id}').offsetWidth - margin.left - margin.right;
                    const height = {height.replace('px', '')} - margin.top - margin.bottom;

                    // Create SVG
                    const svg = d3.select('#{element_id}')
                        .append('svg')
                        .attr('width', width + margin.left + margin.right)
                        .attr('height', height + margin.top + margin.bottom)
                        .append('g')
                        .attr('transform', `translate(${{margin.left}}, ${{margin.top}})`);

                    // X scale
                    const x = d3.scaleBand()
                        .range([0, width])
                        .domain(xLabels)
                        .padding(0.05);

                    // Add X axis
                    svg.append('g')
                        .style('font-size', 10)
                        .attr('transform', `translate(0, ${{height}})`)
                        .call(d3.axisBottom(x).tickSize(0))
                        .selectAll('text')
                        .attr('transform', 'translate(-10,0)rotate(-45)')
                        .style('text-anchor', 'end');

                    // Y scale
                    const y = d3.scaleBand()
                        .range([height, 0])
                        .domain(yLabels)
                        .padding(0.05);

                    // Add Y axis
                    svg.append('g')
                        .style('font-size', 10)
                        .call(d3.axisLeft(y).tickSize(0));

                    // Build color scale
                    const myColor = d3.scaleSequential()
                        .interpolator(d3.interpolateBlues)
                        .domain([0, d3.max(data.flat())]);

                    // Create tooltip div
                    const tooltip = d3.select('body')
                        .append('div')
                        .style('opacity', 0)
                        .attr('class', 'tooltip')
                        .style('background-color', 'white')
                        .style('border', 'solid')
                        .style('border-width', '2px')
                        .style('border-radius', '5px')
                        .style('padding', '5px')
                        .style('position', 'absolute');

                    // Add cells
                    for (let i = 0; i < yLabels.length; i++) {{
                        for (let j = 0; j < xLabels.length; j++) {{
                            svg.append('rect')
                                .attr('x', x(xLabels[j]))
                                .attr('y', y(yLabels[i]))
                                .attr('width', x.bandwidth())
                                .attr('height', y.bandwidth())
                                .style('fill', myColor(data[i][j]))
                                .on('mouseover', function() {{
                                    tooltip.style('opacity', 1);
                                    d3.select(this)
                                        .style('stroke', 'black')
                                        .style('opacity', 1);
                                }})
                                .on('mousemove', function() {{
                                    tooltip
                                        .html(`${{yLabels[i]}} x ${{xLabels[j]}}: ${{data[i][j]}}`)
                                        .style('left', (d3.event.pageX + 10) + 'px')
                                        .style('top', (d3.event.pageY - 10) + 'px');
                                }})
                                .on('mouseleave', function() {{
                                    tooltip.style('opacity', 0);
                                    d3.select(this)
                                        .style('stroke', 'none')
                                        .style('opacity', 0.8);
                                }});
                        }}
                    }}
                }})();
            </script>
            """

        @staticmethod
        def generate_html_report(title: str, charts: List[Tuple[str, Dict[str, Any]]],
                                 summary_text: str = None) -> str:
            """
            Generate a complete HTML report with multiple charts.

            Args:
                title: Report title
                charts: List of tuples (chart_id, chart_config)
                summary_text: Optional summary text in markdown format

            Returns:
                str: Complete HTML report
            """
            chart_html = []
            for chart_id, chart_config in charts:
                chart_html.append(VisualizationRenderer.chart_config_to_html(
                    chart_config, chart_id, "100%", "400px"
                ))

            summary_html = ""
            if summary_text:
                # Simple markdown conversion for summary text
                # For production use, consider a proper markdown library
                summary_lines = []
                for line in summary_text.split('\n'):
                    if line.startswith('# '):
                        summary_lines.append(f"<h1>{line[2:]}</h1>")
                    elif line.startswith('## '):
                        summary_lines.append(f"<h2>{line[3:]}</h2>")
                    elif line.startswith('### '):
                        summary_lines.append(f"<h3>{line[4:]}</h3>")
                    elif line.startswith('- '):
                        summary_lines.append(f"<li>{line[2:]}</li>")
                    else:
                        summary_lines.append(f"<p>{line}</p>")

                summary_html = f"""
                <div class="summary-section">
                    {''.join(summary_lines)}
                </div>
                """

            return f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.6.1/d3.min.js"></script>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .report-header {{
                        text-align: center;
                        margin-bottom: 30px;
                        padding-bottom: 20px;
                        border-bottom: 1px solid #eee;
                    }}
                    .chart-container {{
                        margin-bottom: 40px;
                        padding: 20px;
                        border: 1px solid #eee;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .summary-section {{
                        margin-bottom: 30px;
                        padding: 20px;
                        background-color: #f9f9f9;
                        border-radius: 5px;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-bottom: 20px;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    .tooltip {{
                        position: absolute;
                        background-color: white;
                        border: 1px solid #ddd;
                        padding: 10px;
                        border-radius: 5px;
                        pointer-events: none;
                        z-index: 100;
                    }}
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>{title}</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                {summary_html}

                <div class="charts-section">
                    {"".join([f'<div class="chart-container">{chart}</div>' for chart in chart_html])}
                </div>
            </body>
            </html>
            """

    class DataFormatter:
        """
        Helper class for formatting data to be used in visualizations.
        """

        @staticmethod
        def format_transaction_data_for_timeseries(
                transactions: List[Transaction],
                group_by: str = "day"
        ) -> pd.DataFrame:
            """
            Format transaction data for time series visualization.

            Args:
                transactions: List of Transaction objects
                group_by: Grouping period ('hour', 'day', 'week', 'month')

            Returns:
                pd.DataFrame: Formatted data for visualization
            """
            # Convert to DataFrame
            if not transactions:
                return pd.DataFrame(columns=["date", "buy_volume", "sell_volume"])

            df = pd.DataFrame([{
                "timestamp": tx.timestamp,
                "amount": tx.amount,
                "transaction_type": tx.transaction_type.name if hasattr(tx.transaction_type,
                                                                        'name') else tx.transaction_type
            } for tx in transactions])

            # Convert timestamp to datetime if not already
            if not pd.api.types.is_datetime64_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Create date column based on grouping
            if group_by == "hour":
                df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:00")
            elif group_by == "day":
                df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
            elif group_by == "week":
                df["date"] = df["timestamp"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
            elif group_by == "month":
                df["date"] = df["timestamp"].dt.strftime("%Y-%m")
            else:
                df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

            # Separate buy and sell transactions
            buy_volume = df[df["transaction_type"] == "BUY"].groupby("date")["amount"].sum().reset_index()
            buy_volume.columns = ["date", "buy_volume"]

            sell_volume = df[df["transaction_type"] == "SELL"].groupby("date")["amount"].sum().reset_index()
            sell_volume.columns = ["date", "sell_volume"]

            # Merge buy and sell data
            volume_data = pd.merge(buy_volume, sell_volume, on="date", how="outer").fillna(0)

            # Add total volume
            volume_data["total_volume"] = volume_data["buy_volume"] + volume_data["sell_volume"]

            # Sort by date
            volume_data = volume_data.sort_values("date")

            return volume_data

        @staticmethod
        def format_wallet_data_for_analysis(
                wallet_analyses: List[WalletAnalysis]
        ) -> pd.DataFrame:
            """
            Format wallet analysis data for visualization.

            Args:
                wallet_analyses: List of WalletAnalysis objects

            Returns:
                pd.DataFrame: Formatted data for visualization
            """
            if not wallet_analyses:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([{
                "wallet_address": wa.wallet_address,
                "pair_id": wa.pair_id,
                "total_buy_amount": wa.total_buy_amount,
                "total_sell_amount": wa.total_sell_amount,
                "buy_sell_ratio": wa.buy_sell_ratio if wa.buy_sell_ratio is not None else 0,
                "transaction_count": wa.transaction_count,
                "last_analyzed": wa.last_analyzed
            } for wa in wallet_analyses])

            # Classify wallets based on buy_sell_ratio
            conditions = [
                (df["buy_sell_ratio"] > 1.5),
                (df["buy_sell_ratio"] >= 0.75) & (df["buy_sell_ratio"] <= 1.5),
                (df["buy_sell_ratio"] < 0.75)
            ]

            choices = ["Buyer", "Balanced", "Seller"]
            df["classification"] = np.select(conditions, choices, default="Unknown")

            return df

        @staticmethod
        def format_waves_for_visualization(
                waves: List[Wave]
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """
            Format wave data for visualization.

            Args:
                waves: List of Wave objects

            Returns:
                Tuple[pd.DataFrame, pd.DataFrame]: Buy waves and sell waves DataFrames
            """
            if not waves:
                return pd.DataFrame(), pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([{
                "id": w.id,
                "pair_id": w.pair_id,
                "wave_type": w.wave_type.name if hasattr(w.wave_type, 'name') else w.wave_type,
                "start_timestamp": w.start_timestamp,
                "end_timestamp": w.end_timestamp,
                "total_amount": w.total_amount,
                "transaction_count": w.transaction_count,
                "average_price": w.average_price
            } for w in waves])

            # Split into buy and sell waves
            buy_waves = df[df["wave_type"] == "BUY"].copy()
            sell_waves = df[df["wave_type"] == "SELL"].copy()

            return buy_waves, sell_waves

    class VisualizationExporter:
        """
        Exports visualizations to various formats.
        """

        @staticmethod
        async def export_html_to_file(html_content: str, file_path: str) -> bool:
            """
            Export HTML content to a file.

            Args:
                html_content: HTML content to export
                file_path: Path to save the file

            Returns:
                bool: True if successful, False otherwise
            """
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

                # Write HTML content to file
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(html_content)

                logger.info(f"HTML report exported to {file_path}")
                return True

            except Exception as e:
                logger.error(f"Error exporting HTML report: {str(e)}")
                return False

        @staticmethod
        async def export_chart_as_image(
                chart_config: Dict[str, Any],
                file_path: str,
                width: int = 800,
                height: int = 600
        ) -> bool:
            """
            Export chart as image using external rendering service.

            Args:
                chart_config: Chart configuration
                file_path: Path to save the image
                width: Image width
                height: Image height

            Returns:
                bool: True if successful, False otherwise
            """
            # In a real-world scenario, we would use a library like Selenium or Playwright
            # to render the chart in a headless browser and capture it as an image
            # or use a server-side rendering service like QuickChart.io

            logger.warning("Chart image export is not implemented yet. Requires headless browser or rendering service.")
            return False
