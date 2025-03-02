# src/reports/exporter.py
import os
import json
import csv
import asyncio
import aiofiles
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
from loguru import logger
import pandas as pd
import jinja2
import pdfkit

from src.reports.report_generator import ReportGenerator
from src.reports.visualization import VisualizationRenderer


class ReportExporter:
    """
    Handles exporting reports to various formats (JSON, CSV, HTML, PDF, etc.)
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report exporter with a default output directory.

        Args:
            output_dir: Directory to store exported reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set up Jinja2 environment for templating
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("templates"),
            autoescape=True
        )

    async def export_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export report data to JSON format.

        Args:
            data: Report data to export
            filename: Output filename (without extension)

        Returns:
            str: Path to the exported file
        """
        file_path = os.path.join(self.output_dir, f"{filename}.json")

        # Make data JSON serializable
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)

        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                serialized_data = json.dumps(data, default=serialize, indent=4)
                await f.write(serialized_data)

            logger.info(f"Exported JSON report to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise

    async def export_to_csv(self, data: Union[List[Dict], pd.DataFrame], filename: str) -> str:
        """
        Export tabular data to CSV format.

        Args:
            data: List of dictionaries or pandas DataFrame to export
            filename: Output filename (without extension)

        Returns:
            str: Path to the exported file
        """
        file_path = os.path.join(self.output_dir, f"{filename}.csv")

        try:
            # Convert to DataFrame if not already
            if not isinstance(data, pd.DataFrame):
                df = pd.DataFrame(data)
            else:
                df = data

            # Use pandas to write CSV asynchronously
            await asyncio.to_thread(df.to_csv, file_path, index=False)

            logger.info(f"Exported CSV report to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise

    async def export_to_html(self,
                             report_data: Dict[str, Any],
                             template_name: str,
                             filename: str,
                             charts: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Export report data to HTML using a template.

        Args:
            report_data: Report data to include in the template
            template_name: Name of the Jinja2 template file
            filename: Output filename (without extension)
            charts: Optional list of chart configurations to render

        Returns:
            str: Path to the exported file
        """
        file_path = os.path.join(self.output_dir, f"{filename}.html")

        try:
            # Get the template
            template = self.jinja_env.get_template(f"{template_name}.html")

            # Generate charts HTML if provided
            charts_html = ""
            if charts:
                for i, chart_config in enumerate(charts):
                    chart_id = f"chart_{i}"
                    chart_html = VisualizationRenderer.chart_config_to_html(
                        chart_config, chart_id, "100%", "400px"
                    )
                    charts_html += f'<div class="chart-container">{chart_html}</div>'

            # Render the template with data
            html_content = template.render(
                report=report_data,
                charts_html=charts_html,
                generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

            # Write to file
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(html_content)

            logger.info(f"Exported HTML report to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error exporting to HTML: {str(e)}")
            raise

    async def export_to_pdf(self, html_path: str, filename: str) -> str:
        """
        Convert HTML report to PDF.

        Args:
            html_path: Path to the HTML file
            filename: Output filename (without extension)

        Returns:
            str: Path to the exported PDF file
        """
        pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")

        try:
            # Use pdfkit (wkhtmltopdf wrapper) to convert HTML to PDF
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': 'UTF-8',
                'no-outline': None,
                'enable-local-file-access': None
            }

            # Use asyncio to run this blocking operation in a thread pool
            await asyncio.to_thread(
                pdfkit.from_file, html_path, pdf_path, options=options
            )

            logger.info(f"Exported PDF report to {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"Error exporting to PDF: {str(e)}")
            logger.warning("Make sure wkhtmltopdf is installed on your system")
            raise

    async def export_chart_image(self,
                                 chart_config: Dict[str, Any],
                                 filename: str,
                                 width: int = 800,
                                 height: int = 600) -> str:
        """
        Export a chart as an image file (PNG).
        This is a placeholder that would use a headless browser or service in production.

        Args:
            chart_config: Chart configuration
            filename: Output filename (without extension)
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            str: Path to the exported image
        """
        # Generate HTML with just this chart
        chart_html = VisualizationRenderer.chart_config_to_html(
            chart_config, "chart", "100%", "100%"
        )

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.6.1/d3.min.js"></script>
            <style>
                body {{ margin: 0; padding: 0; }}
                #chart {{ width: {width}px; height: {height}px; }}
            </style>
        </head>
        <body>
            {chart_html}
        </body>
        </html>
        """

        # Save HTML file
        temp_html_path = os.path.join(self.output_dir, f"{filename}_temp.html")
        output_path = os.path.join(self.output_dir, f"{filename}.png")

        try:
            # Write HTML to temporary file
            async with aiofiles.open(temp_html_path, 'w', encoding='utf-8') as f:
                await f.write(html_content)

            logger.warning("Chart image export would require headless browser rendering")
            logger.info(f"Created HTML version at {temp_html_path}")
            logger.info(f"To implement actual image export, use Selenium, Playwright, or a service like QuickChart.io")

            # In a real implementation, we would use something like:
            # from playwright.async_api import async_playwright
            # async with async_playwright() as p:
            #     browser = await p.chromium.launch()
            #     page = await browser.new_page(viewport={"width": width, "height": height})
            #     await page.goto(f"file://{temp_html_path}")
            #     await page.wait_for_timeout(1000)  # Wait for chart rendering
            #     await page.screenshot(path=output_path)
            #     await browser.close()

            return temp_html_path  # In reality, would return the PNG path

        except Exception as e:
            logger.error(f"Error exporting chart to image: {str(e)}")
            raise
