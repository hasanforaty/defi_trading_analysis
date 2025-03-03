# src/cli/report_cli.py
import asyncio
import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger

from src.data.db_manager import DatabaseManager
from src.models.entities import Transaction, WalletAnalysis, Wave, Pair
from src.analyzers.threshold import ThresholdAnalyzer
from src.analyzers.wave import WaveDetector
from src.analyzers.ratio import RatioAnalyzer
from src.analyzers.pattern import PatternRecognizer
from src.analyzers.coordinator import AnalysisCoordinator
from src.reports.generator import (
    ReportGenerator, TransactionReport, WaveReport,
    WalletAnalysisReport, ComprehensiveReport
)
from src.reports.visualization import ChartGenerator, VisualizationRenderer
from src.reports.exporter import ReportExporter


class ReportingCLI:
    """
    Command-line interface for generating reports from analyzed DeFi trading data.
    """

    def __init__(self):
        """Initialize the CLI parser and commands."""
        self.db_manager = DatabaseManager()
        self.parser = argparse.ArgumentParser(
            description="DeFi Trading Pattern Analysis Reporting Tool",
            formatter_class=argparse.RawTextHelpFormatter
        )

        self._setup_parser()

    def _setup_parser(self):
        """Set up the command-line argument parser."""
        # Add global options
        self.parser.add_argument(
            "--output-dir",
            default="reports",
            help="Directory to store generated reports"
        )
        self.parser.add_argument(
            "--format",
            choices=["json", "csv", "html", "pdf"],
            default="html",
            help="Output format for the report"
        )

        # Create subparsers for different report types
        subparsers = self.parser.add_subparsers(
            dest="command",
            title="Available Commands",
            help="Type of report to generate"
        )

        # Transaction report command
        tx_parser = subparsers.add_parser(
            "transactions",
            help="Generate transaction analysis report"
        )
        tx_parser.add_argument(
            "--pair-id",
            type=int,
            required=True,
            help="ID of the trading pair to analyze"
        )
        tx_parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Number of days to look back for transactions"
        )
        tx_parser.add_argument(
            "--min-amount",
            type=float,
            help="Minimum transaction amount to include"
        )
        tx_parser.add_argument(
            "--visualize",
            action="store_true",
            help="Include visualizations in the report"
        )

        # Wave report command
        wave_parser = subparsers.add_parser(
            "waves",
            help="Generate wave analysis report"
        )
        wave_parser.add_argument(
            "--pair-id",
            type=int,
            required=True,
            help="ID of the trading pair to analyze"
        )
        wave_parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Number of days to look back for waves"
        )
        wave_parser.add_argument(
            "--min-amount",
            type=float,
            default=0,
            help="Minimum wave amount to include"
        )
        wave_parser.add_argument(
            "--min-transactions",
            type=int,
            default=0,
            help="Minimum transactions in a wave"
        )
        wave_parser.add_argument(
            "--visualize",
            action="store_true",
            help="Include visualizations in the report"
        )

        # Wallet analysis report command
        wallet_parser = subparsers.add_parser(
            "wallets",
            help="Generate wallet analysis report"
        )
        wallet_parser.add_argument(
            "--pair-id",
            type=int,
            default=None,
            help="Optional ID of the trading pair to filter by"
        )
        wallet_parser.add_argument(
            "--wallet",
            type=str,
            default=None,
            help="Optional wallet address to analyze"
        )
        wallet_parser.add_argument(
            "--top",
            type=int,
            default=100,
            help="Number of top wallets to include"
        )
        wallet_parser.add_argument(
            "--min-ratio",
            type=float,
            default=None,
            help="Minimum buy/sell ratio to include"
        )
        wallet_parser.add_argument(
            "--max-ratio",
            type=float,
            default=None,
            help="Maximum buy/sell ratio to include"
        )
        wallet_parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Number of days to look back"
        )
        wallet_parser.add_argument(
            "--visualize",
            action="store_true",
            help="Include visualizations in the report"
        )

        # Comprehensive report command
        comp_parser = subparsers.add_parser(
            "comprehensive",
            help="Generate comprehensive analysis report"
        )
        comp_parser.add_argument(
            "--pair-id",
            type=int,
            required=True,
            help="ID of the trading pair to analyze"
        )
        comp_parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Number of days to look back"
        )
        comp_parser.add_argument(
            "--threshold-multiplier",
            type=float,
            default=2.0,
            help="Multiplier for transaction threshold calculation"
        )
        comp_parser.add_argument(
            "--min-wave-amount",
            type=float,
            default=1000.0,
            help="Minimum amount for wave detection"
        )
        comp_parser.add_argument(
            "--min-ratio",
            type=float,
            default=0.7,
            help="Minimum buy/sell ratio for 'buyer' classification"
        )
        comp_parser.add_argument(
            "--include-tables",
            action="store_true",
            help="Include detailed data tables in the report"
        )
        comp_parser.add_argument(
            "--visualize",
            action="store_true",
            help="Include visualizations in the report"
        )

    async def _generate_transaction_report(self, args):
        """Generate transaction analysis report."""
        async with self.db_manager.session() as session:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=args.days)

            # Create report generator
            report_gen = TransactionReport(session)

            # Generate report
            report_data = await report_gen.generate_report(
                pair_id=args.pair_id,
                start_time=start_date,
                end_time=end_date,
                min_amount=args.min_amount,
                include_charts=args.visualize
            )

            # Create exporter
            exporter = ReportExporter(args.output_dir)

            # Export report in requested format
            filename = f"transaction_report_pair_{args.pair_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if args.format == "json":
                path = await exporter.export_to_json(report_data, filename)
            elif args.format == "csv":
                if "transactions" in report_data:
                    path = await exporter.export_to_csv(report_data["transactions"], filename)
                else:
                    logger.warning("No transaction data available for CSV export")
                    path = await exporter.export_to_json(report_data, filename)
            elif args.format == "html":
                charts = []
                if args.visualize and "charts" in report_data:
                    charts = report_data["charts"]

                path = await exporter.export_to_html(
                    report_data=report_data,
                    template_name="transaction_report",
                    filename=filename,
                    charts=charts
                )
            elif args.format == "pdf":
                # First export to HTML, then convert to PDF
                html_path = await exporter.export_to_html(
                    report_data=report_data,
                    template_name="transaction_report",
                    filename=f"{filename}_temp",
                    charts=report_data.get("charts", []) if args.visualize else []
                )
                path = await exporter.export_to_pdf(html_path, filename)

            logger.info(f"Transaction report generated successfully: {path}")
            return path

    async def _generate_wave_report(self, args):
        """Generate wave analysis report."""
        async with self.db_manager.session() as session:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=args.days)

            # Create report generator
            report_gen = WaveReport(session)

            # Generate report
            report_data = await report_gen.generate_report(
                pair_id=args.pair_id,
                start_time=start_date,
                end_time=end_date,
                min_amount=args.min_amount,
                min_transactions=args.min_transactions,
                include_charts=args.visualize
            )

            # Create exporter
            exporter = ReportExporter(args.output_dir)

            # Export report in requested format
            filename = f"wave_report_pair_{args.pair_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if args.format == "json":
                path = await exporter.export_to_json(report_data, filename)
            elif args.format == "csv":
                if "waves" in report_data:
                    # Combine buy and sell waves
                    all_waves = report_data.get("buy_waves", []) + report_data.get("sell_waves", [])
                    path = await exporter.export_to_csv(all_waves, filename)
                else:
                    logger.warning("No wave data available for CSV export")
                    path = await exporter.export_to_json(report_data, filename)
            elif args.format == "html":
                charts = []
                if args.visualize and "charts" in report_data:
                    charts = report_data["charts"]

                path = await exporter.export_to_html(
                    report_data=report_data,
                    template_name="wave_report",
                    filename=filename,
                    charts=charts
                )
            elif args.format == "pdf":
                # First export to HTML, then convert to PDF
                html_path = await exporter.export_to_html(
                    report_data=report_data,
                    template_name="wave_report",
                    filename=f"{filename}_temp",
                    charts=report_data.get("charts", []) if args.visualize else []
                )
                path = await exporter.export_to_pdf(html_path, filename)

            logger.info(f"Wave report generated successfully: {path}")
            return path

    async def _generate_wallet_report(self, args):
        """Generate wallet analysis report."""
        async with self.db_manager.session() as session:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=args.days)

            # Create report generator
            report_gen = WalletAnalysisReport(session)

            # Generate report
            report_data = await report_gen.generate_report(
                wallet_address=args.wallet,
                pair_id=args.pair_id,
                start_time=start_date,
                end_time=end_date,
                min_ratio=args.min_ratio,
                max_ratio=args.max_ratio,
                top_n=args.top,
                include_charts=args.visualize
            )

            # Create exporter
            exporter = ReportExporter(args.output_dir)

            # Export report in requested format
            if args.wallet:
                filename = f"wallet_report_{args.wallet[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            elif args.pair_id:
                filename = f"wallet_report_pair_{args.pair_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                filename = f"wallet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if args.format == "json":
                path = await exporter.export_to_json(report_data, filename)
            elif args.format == "csv":
                if "wallets" in report_data:
                    path = await exporter.export_to_csv(report_data["wallets"], filename)
                else:
                    logger.warning("No wallet data available for CSV export")
                    path = await exporter.export_to_json(report_data, filename)
            elif args.format == "html":
                charts = []
                if args.visualize and "charts" in report_data:
                    charts = report_data["charts"]

                path = await exporter.export_to_html(
                    report_data=report_data,
                    template_name="wallet_report",
                    filename=filename,
                    charts=charts
                )
            elif args.format == "pdf":
                # First export to HTML, then convert to PDF
                html_path = await exporter.export_to_html(
                    report_data=report_data,
                    template_name="wallet_report",
                    filename=f"{filename}_temp",
                    charts=report_data.get("charts", []) if args.visualize else []
                )
                path = await exporter.export_to_pdf(html_path, filename)

            logger.info(f"Wallet analysis report generated successfully: {path}")
            return path

    async def _generate_comprehensive_report(self, args):
        """Generate comprehensive analysis report."""
        async with self.db_manager.session() as session:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=args.days)

            # Create report generator
            report_gen = ComprehensiveReport(session)

            # Generate report
            report_data = await report_gen.generate_report(
                pair_id=args.pair_id,
                start_time=start_date,
                end_time=end_date,
                threshold_multiplier=args.threshold_multiplier,
                min_wave_amount=args.min_wave_amount,
                min_ratio=args.min_ratio,
                include_tables=args.include_tables,
                include_charts=args.visualize
            )

            # Create exporter
            exporter = ReportExporter(args.output_dir)

            # Export report in requested format
            filename = f"comprehensive_report_pair_{args.pair_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if args.format == "json":
                path = await exporter.export_to_json(report_data, filename)
            elif args.format == "csv":
                logger.warning("Comprehensive report contains multiple datasets, exporting as JSON instead")
                path = await exporter.export_to_json(report_data, filename)
            elif args.format == "html":
                charts = []
                if args.visualize and "charts" in report_data:
                    charts = report_data["charts"]

                path = await exporter.export_to_html(
                    report_data=report_data,
                    template_name="comprehensive_report",
                    filename=filename,
                    charts=charts
                )
            elif args.format == "pdf":
                # First export to HTML, then convert to PDF
                html_path = await exporter.export_to_html(
                    report_data=report_data,
                    template_name="comprehensive_report",
                    filename=f"{filename}_temp",
                    charts=report_data.get("charts", []) if args.visualize else []
                )
                path = await exporter.export_to_pdf(html_path, filename)

            logger.info(f"Comprehensive report generated successfully: {path}")
            return path

    async def run(self, args=None):
        """Run the CLI with the given arguments."""
        if args is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(args)

        # Initialize database connection
        await self.db_manager.init()

        try:
            if args.command == "transactions":
                return await self._generate_transaction_report(args)
            elif args.command == "waves":
                return await self._generate_wave_report(args)
            elif args.command == "wallets":
                return await self._generate_wallet_report(args)
            elif args.command == "comprehensive":
                return await self._generate_comprehensive_report(args)
            else:
                self.parser.print_help()
                return None

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
        finally:
            # Close database connection
            await self.db_manager.close()


def main():
    """Entry point for the CLI."""
    cli = ReportingCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
