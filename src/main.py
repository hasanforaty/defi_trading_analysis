# src/main.py
import asyncio
import argparse
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger

from src.data.db_manager import DatabaseManager
from src.analyzers.cordinator import AnalysisCoordinator
from src.cli.report_cli import ReportingCLI
from src.models.entities import Pair


class DeFiTradingAnalysisApp:
    """
    Main application entry point for DeFi Trading Pattern Analysis.
    Integrates all components and provides a unified interface.
    """

    def __init__(self):
        """Initialize the application."""
        self.db_manager = DatabaseManager()
        self.parser = argparse.ArgumentParser(
            description="DeFi Trading Pattern Analysis Tool",
            formatter_class=argparse.RawTextHelpFormatter
        )
        self._setup_parser()

    def _setup_parser(self):
        """Set up the command-line argument parser."""
        # Add global options
        self.parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        self.parser.add_argument("--log-file", help="Log file path")

        # Create subparsers for different commands
        subparsers = self.parser.add_subparsers(
            dest="command",
            title="Available Commands",
            help="Command to execute"
        )

        # Analyze command
        analyze_parser = subparsers.add_parser(
            "analyze",
            help="Run analysis on trading pairs"
        )
        analyze_parser.add_argument(
            "--pair-id",
            type=int,
            required=True,
            help="ID of the trading pair to analyze"
        )
        analyze_parser.add_argument(
            "--type",
            choices=["threshold", "wave", "ratio", "pattern", "comprehensive"],
            default="comprehensive",
            help="Type of analysis to run"
        )
        analyze_parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Number of days to look back"
        )
        analyze_parser.add_argument(
            "--wait",
            action="store_true",
            help="Wait for analysis to complete instead of running in background"
        )

        # Report command (forwards to report CLI)
        report_parser = subparsers.add_parser(
            "report",
            help="Generate reports from analysis data"
        )
        # We'll forward all arguments to the report CLI
        report_parser.add_argument(
            "report_args",
            nargs=argparse.REMAINDER,
            help="Arguments to pass to the report CLI"
        )

        # List command
        list_parser = subparsers.add_parser(
            "list",
            help="List available data"
        )
        list_parser.add_argument(
            "--type",
            choices=["pairs", "jobs", "reports"],
            required=True,
            help="Type of data to list"
        )
        list_parser.add_argument(
            "--limit",
            type=int,
            default=20,
            help="Maximum number of items to show"
        )

    async def _configure_logging(self, debug: bool, log_file: Optional[str] = None):
        """Configure application logging."""
        log_level = logging.DEBUG if debug else logging.INFO

        # Configure console logging
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level=log_level)

        # Configure file logging if specified
        if log_file:
            logger.add(log_file, rotation="10 MB", retention="1 week", level=log_level)

        logger.info(f"Logging initialized at level: {'DEBUG' if debug else 'INFO'}")

    async def _run_analysis(self, args):
        """Run analysis based on command-line arguments."""
        async with self.db_manager.session() as session:
            coordinator = AnalysisCoordinator(session)

            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=args.days)

            # Execute the requested analysis type
            if args.type == "threshold":
                job_id = await coordinator.run_threshold_analysis(
                    pair_id=args.pair_id,
                    start_time=start_time,
                    end_time=end_time
                )
            elif args.type == "wave":
                job_id = await coordinator.run_wave_detection(
                    pair_id=args.pair_id,
                    start_time=start_time,
                    end_time=end_time
                )
            elif args.type == "ratio":
                job_id = await coordinator.run_ratio_analysis(
                    pair_id=args.pair_id,
                    days_lookback=args.days
                )
            elif args.type == "pattern":
                job_id = await coordinator.run_pattern_recognition(
                    pair_id=args.pair_id,
                    start_time=start_time,
                    end_time=end_time
                )
            else:  # comprehensive
                job_id = await coordinator.run_comprehensive_analysis(
                    pair_id=args.pair_id,
                    days_lookback=args.days
                )

            logger.info(f"Analysis job started with ID: {job_id}")

            # Wait for job to complete if requested
            if args.wait:
                logger.info("Waiting for analysis to complete...")

                # Poll job status until done
                while True:
                    status = await coordinator.get_job_status(job_id)
                    if status["status"] in ["completed", "failed"]:
                        break

                    # Show progress
                    logger.info(f"Progress: {status['progress']:.1f}%")
                    await asyncio.sleep(1)

                # Show job result
                final_status = await coordinator.get_job_status(job_id)
                if final_status["status"] == "completed":
                    logger.info(f"Analysis completed successfully in {final_status['duration']:.2f} seconds")
                    result = await coordinator.get_job_result(job_id)
                    logger.info(
                        f"Result summary: {', '.join([f'{k}: {v}' for k, v in result.items() if not isinstance(v, dict)])}")
                else:
                    logger.error(f"Analysis failed: {', '.join(final_status.get('errors', ['Unknown error']))}")

            return job_id

    async def _run_report(self, args):
        """Run report generation by forwarding to the ReportingCLI."""
        reporting_cli = ReportingCLI()
        return await reporting_cli.run(args.report_args)

    async def _list_data(self, args):
        """List available data based on the specified type."""
        async with self.db_manager.session() as session:
            if args.type == "pairs":
                # List available trading pairs
                result = await session.execute(
                    "SELECT id, token0_symbol, token1_symbol, address, chain FROM pairs LIMIT :limit",
                    {"limit": args.limit})
                pairs = result.mappings().all()

                print("\nAvailable Trading Pairs:")
                print("------------------------")
                print(f"{'ID':<5} {'Token Pair':<15} {'Address':<42} {'Chain':<10}")
                print("-" * 80)

                for pair in pairs:
                    token_pair = f"{pair['token0_symbol']}/{pair['token1_symbol']}"
                    print(f"{pair['id']:<5} {token_pair:<15} {pair['address']:<42} {pair['chain']:<10}")

                print(f"\nTotal: {len(pairs)} pairs")

            elif args.type == "jobs":
                # List recent analysis jobs
                coordinator = AnalysisCoordinator(session)

                # Get active jobs
                active_jobs = await coordinator.get_active_jobs()

                # Get completed jobs
                completed_jobs = await coordinator.get_completed_jobs(limit=args.limit - len(active_jobs))

                # Display active jobs
                print("\nActive Analysis Jobs:")
                print("--------------------")
                if active_jobs:
                    print(f"{'Job ID':<36} {'Type':<15} {'Pair ID':<8} {'Progress':<10} {'Started':<20}")
                    print("-" * 90)

                    for job in active_jobs:
                        print(
                            f"{job['job_id']:<36} {job['job_type']:<15} {job['pair_id']:<8} {job['progress']:.1f}% {job['start_time'][:19]:<20}")
                else:
                    print("No active jobs\n")

                # Display completed jobs
                print("\nCompleted Analysis Jobs:")
                print("-----------------------")
                if completed_jobs:
                    print(f"{'Job ID':<36} {'Type':<15} {'Pair ID':<8} {'Status':<10} {'Duration':<10}")
                    print("-" * 90)

                    for job in completed_jobs:
                        duration = f"{job['duration']:.2f}s" if job['duration'] else "N/A"
                        print(
                            f"{job['job_id']:<36} {job['job_type']:<15} {job['pair_id']:<8} {job['status']:<10} {duration:<10}")
                else:
                    print("No completed jobs\n")

            elif args.type == "reports":
                # List available reports in the reports directory
                reports_dir = Path("reports")
                if not reports_dir.exists():
                    print("\nNo reports directory found")
                    return

                # Get all report files
                report_files = list(reports_dir.glob("*.html")) + list(reports_dir.glob("*.pdf")) + list(
                    reports_dir.glob("*.json"))
                report_files = sorted(report_files, key=lambda x: x.stat().st_mtime, reverse=True)[:args.limit]

                print("\nAvailable Reports:")
                print("-----------------")
                print(f"{'Filename':<50} {'Type':<10} {'Size':<10} {'Created':<20}")
                print("-" * 90)

                for report_file in report_files:
                    file_type = report_file.suffix[1:]
                    size_kb = report_file.stat().st_size / 1024
                    modified_time = datetime.fromtimestamp(report_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                    print(f"{report_file.name:<50} {file_type:<10} {size_kb:.1f} KB {modified_time:<20}")

                print(f"\nTotal: {len(report_files)} reports")

    async def run(self, args=None):
        """Run the application with the given arguments."""
        if args is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(args)

        # Configure logging
        await self._configure_logging(args.debug, args.log_file)

        # Initialize database connection
        await self.db_manager.init()
        logger.info("Database connection initialized")

        try:
            # Execute the requested command
            if args.command == "analyze":
                return await self._run_analysis(args)
            elif args.command == "report":
                return await self._run_report(args)
            elif args.command == "list":
                return await self._list_data(args)
            else:
                self.parser.print_help()
                return None

        except Exception as e:
            logger.exception(f"Error executing command: {str(e)}")
            raise
        finally:
            # Close database connection
            await self.db_manager.close()
            logger.info("Database connection closed")


def main():
    """Entry point for the application."""
    app = DeFiTradingAnalysisApp()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()

