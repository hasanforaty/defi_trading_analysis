# src/reports/generator.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import pandas as pd
from pathlib import Path
import json
import asyncio
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from config.settings import get_settings


class ReportGenerator(ABC):
    """
    Abstract base class for different report types.
    Provides common structure and methods for all report generators.
    """

    def __init__(
            self,
            session: AsyncSession,
            title: str = None,
            description: str = None,
            config: Dict[str, Any] = None
    ):
        """
        Initialize the report generator.

        Args:
            session: Database session
            title: Report title
            description: Report description
            config: Configuration options
        """
        self.session = session
        self.title = title or self.__class__.__name__
        self.description = description or f"Report generated at {datetime.utcnow().isoformat()}"
        self.config = config or {}
        self.settings = get_settings()
        self.data_sources = {}
        self.report_sections = []
        self.progress_callback = None
        self.progress = 0.0
        self.metadata = {
            "generated_at": datetime.utcnow().isoformat(),
            "report_type": self.__class__.__name__,
        }

    async def set_data_source(self, name: str, data: Any) -> None:
        """
        Set a data source for the report.

        Args:
            name: Name of the data source
            data: Data source content
        """
        self.data_sources[name] = data

    async def load_data_source(self, name: str, loader_func: Callable) -> None:
        """
        Load a data source using a loader function.

        Args:
            name: Name of the data source
            loader_func: Async function that returns data
        """
        data = await loader_func()
        await self.set_data_source(name, data)

    def register_progress_callback(self, callback: Callable[[float], None]) -> None:
        """
        Register a callback function for progress updates.

        Args:
            callback: Function to call with progress percentage (0-100)
        """
        self.progress_callback = callback

    def update_progress(self, progress: float) -> None:
        """
        Update the current progress and trigger the callback if registered.

        Args:
            progress: Progress percentage (0-100)
        """
        self.progress = min(max(progress, 0.0), 100.0)
        if self.progress_callback:
            self.progress_callback(self.progress)

    def add_section(self, title: str, content: Any, section_type: str = "text") -> None:
        """
        Add a section to the report.

        Args:
            title: Section title
            content: Section content
            section_type: Type of section (text, table, chart, etc.)
        """
        self.report_sections.append({
            "title": title,
            "content": content,
            "type": section_type
        })

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report to a dictionary.

        Returns:
            Dictionary representation of the report
        """
        return {
            "title": self.title,
            "description": self.description,
            "metadata": self.metadata,
            "sections": self.report_sections,
            "config": self.config
        }

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert report sections to pandas DataFrames where applicable.

        Returns:
            Dictionary of section titles to DataFrames
        """
        dfs = {}
        for section in self.report_sections:
            if section["type"] == "table" and isinstance(section["content"], (list, dict)):
                # Convert list of dicts to DataFrame
                if isinstance(section["content"], list) and all(isinstance(item, dict) for item in section["content"]):
                    dfs[section["title"]] = pd.DataFrame(section["content"])
                # Convert dict of records to DataFrame
                elif isinstance(section["content"], dict) and "records" in section["content"]:
                    dfs[section["title"]] = pd.DataFrame(section["content"]["records"])
            elif isinstance(section["content"], pd.DataFrame):
                dfs[section["title"]] = section["content"]

        return dfs

    def get_section(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get a section by title.

        Args:
            title: Section title

        Returns:
            Section dictionary or None if not found
        """
        for section in self.report_sections:
            if section["title"] == title:
                return section
        return None

    def get_section_content(self, title: str) -> Optional[Any]:
        """
        Get section content by title.

        Args:
            title: Section title

        Returns:
            Section content or None if not found
        """
        section = self.get_section(title)
        return section["content"] if section else None

    @abstractmethod
    async def generate(self) -> Dict[str, Any]:
        """
        Generate the report. Must be implemented by subclasses.

        Returns:
            Generated report as a dictionary
        """
        pass

    async def apply_template(self, template_name: str) -> str:
        """
        Apply a template to the report.

        Args:
            template_name: Name of the template

        Returns:
            Formatted report as string
        """
        # This will be implemented in a more advanced way later
        return json.dumps(self.to_dict(), indent=2)

    def _convert_data_for_export(self) -> Dict[str, Any]:
        """
        Convert internal data to a format suitable for export.

        Returns:
            Export-ready data
        """
        export_data = self.to_dict()

        # Convert any complex types to serializable formats
        for section in export_data["sections"]:
            if isinstance(section["content"], pd.DataFrame):
                section["content"] = section["content"].to_dict(orient="records")

        return export_data


class TransactionReport(ReportGenerator):
    """
    Report generator for transaction analysis.
    Focuses on significant transactions, volume over time, and wallet activity.
    """

    async def generate(self) -> Dict[str, Any]:
        """
        Generate a transaction report.

        Returns:
            Generated report as a dictionary
        """
        if not self.data_sources:
            raise ValueError("No data sources provided for report generation")

        self.update_progress(10.0)

        # Process significant transactions if available
        if "significant_transactions" in self.data_sources:
            transactions = self.data_sources["significant_transactions"]

            # Convert to DataFrame for easier manipulation
            if not isinstance(transactions, pd.DataFrame):
                transactions_df = pd.DataFrame(transactions)
            else:
                transactions_df = transactions

            # Add the summary section
            self.add_section(
                "Transaction Summary",
                {
                    "total_transactions": len(transactions_df),
                    "total_volume": transactions_df["amount"].sum() if "amount" in transactions_df.columns else 0,
                    "unique_wallets": transactions_df[
                        "wallet_address"].nunique() if "wallet_address" in transactions_df.columns else 0,
                    "average_transaction_size": transactions_df[
                        "amount"].mean() if "amount" in transactions_df.columns else 0,
                },
                "summary"
            )

            self.update_progress(40.0)

            # Add transactions table
            if not transactions_df.empty:
                self.add_section(
                    "Significant Transactions",
                    transactions_df,
                    "table"
                )

            # Process transaction volume over time
            if "timestamp" in transactions_df.columns and "amount" in transactions_df.columns:
                transactions_df["date"] = pd.to_datetime(transactions_df["timestamp"]).dt.date
                volume_by_day = transactions_df.groupby("date")["amount"].sum().reset_index()

                self.add_section(
                    "Transaction Volume Over Time",
                    volume_by_day,
                    "timeseries"
                )

            self.update_progress(70.0)

            # Process top wallets by transaction count and volume
            if "wallet_address" in transactions_df.columns:
                wallet_stats = transactions_df.groupby("wallet_address").agg({
                    "amount": ["sum", "count", "mean", "max"],
                }).reset_index()

                wallet_stats.columns = ["wallet_address", "total_volume", "transaction_count", "average_transaction",
                                        "largest_transaction"]
                wallet_stats = wallet_stats.sort_values("total_volume", ascending=False).head(20)

                self.add_section(
                    "Top Wallets by Volume",
                    wallet_stats,
                    "table"
                )

        self.update_progress(100.0)

        return self.to_dict()


class WaveReport(ReportGenerator):
    """
    Report generator for wave analysis.
    Focuses on detected buy/sell waves, wave patterns, and timeline visualization.
    """

    async def generate(self) -> Dict[str, Any]:
        """
        Generate a wave report.

        Returns:
            Generated report as a dictionary
        """
        if not self.data_sources:
            raise ValueError("No data sources provided for report generation")

        self.update_progress(10.0)

        # Process buy waves if available
        if "buy_waves" in self.data_sources:
            buy_waves = self.data_sources["buy_waves"]


            # Convert to DataFrame for easier manipulation
            if not isinstance(buy_waves, pd.DataFrame):
                buy_waves_df = pd.DataFrame(buy_waves)
            else:
                buy_waves_df = buy_waves

            if not buy_waves_df.empty:
                # Add buy waves summary
                self.add_section(
                    "Buy Waves Summary",
                    {
                        "total_waves": len(buy_waves_df),
                        "total_volume": buy_waves_df[
                            "total_amount"].sum() if "total_amount" in buy_waves_df.columns else 0,
                        "average_wave_size": buy_waves_df[
                            "total_amount"].mean() if "total_amount" in buy_waves_df.columns else 0,
                        "average_transactions_per_wave": buy_waves_df[
                            "transaction_count"].mean() if "transaction_count" in buy_waves_df.columns else 0,
                    },
                    "summary"
                )

                # Add buy waves table
                self.add_section(
                    "Buy Waves",
                    buy_waves_df,
                    "table"
                )

        self.update_progress(40.0)

        # Process sell waves if available
        if "sell_waves" in self.data_sources:
            sell_waves = self.data_sources["sell_waves"]

            # Convert to DataFrame for easier manipulation
            if not isinstance(sell_waves, pd.DataFrame):
                sell_waves_df = pd.DataFrame(sell_waves)
            else:
                sell_waves_df = sell_waves

            if not sell_waves_df.empty:
                # Add sell waves summary
                self.add_section(
                    "Sell Waves Summary",
                    {
                        "total_waves": len(sell_waves_df),
                        "total_volume": sell_waves_df[
                            "total_amount"].sum() if "total_amount" in sell_waves_df.columns else 0,
                        "average_wave_size": sell_waves_df[
                            "total_amount"].mean() if "total_amount" in sell_waves_df.columns else 0,
                        "average_transactions_per_wave": sell_waves_df[
                            "transaction_count"].mean() if "transaction_count" in sell_waves_df.columns else 0,
                    },
                    "summary"
                )

                # Add sell waves table
                self.add_section(
                    "Sell Waves",
                    sell_waves_df,
                    "table"
                )

        self.update_progress(70.0)

        # Combine buy and sell waves for timeline visualization
        if "buy_waves" in self.data_sources and "sell_waves" in self.data_sources:
            buy_waves_df = pd.DataFrame(self.data_sources["buy_waves"]) if not isinstance(
                self.data_sources["buy_waves"], pd.DataFrame) else self.data_sources["buy_waves"]
            sell_waves_df = pd.DataFrame(self.data_sources["sell_waves"]) if not isinstance(
                self.data_sources["sell_waves"], pd.DataFrame) else self.data_sources["sell_waves"]

            # Add wave type column for identification
            if not buy_waves_df.empty:
                buy_waves_df["wave_type"] = "BUY"

            if not sell_waves_df.empty:
                sell_waves_df["wave_type"] = "SELL"

            # Combine and sort by timestamp
            if not buy_waves_df.empty or not sell_waves_df.empty:
                all_waves_df = pd.concat([buy_waves_df, sell_waves_df])

                if not all_waves_df.empty and "start_timestamp" in all_waves_df.columns:
                    all_waves_df["start_date"] = pd.to_datetime(all_waves_df["start_timestamp"]).dt.date
                    all_waves_df = all_waves_df.sort_values("start_timestamp")

                    # Add timeline section
                    self.add_section(
                        "Wave Timeline",
                        all_waves_df,
                        "timeline"
                    )

        self.update_progress(100.0)

        return self.to_dict()


class WalletAnalysisReport(ReportGenerator):
    """
    Report generator for wallet analysis.
    Focuses on buy/sell ratios, wallet behavior classification, and transaction history.
    """

    async def generate(self) -> Dict[str, Any]:
        """
        Generate a wallet analysis report.

        Returns:
            Generated report as a dictionary
        """
        if not self.data_sources:
            raise ValueError("No data sources provided for report generation")

        self.update_progress(10.0)

        # Process wallet analysis data
        if "wallet_analysis" in self.data_sources:
            wallet_data = self.data_sources["wallet_analysis"]

            # Convert to DataFrame for easier manipulation
            if not isinstance(wallet_data, pd.DataFrame):
                wallet_df = pd.DataFrame(wallet_data)
            else:
                wallet_df = wallet_data

            if not wallet_df.empty:
                # Add wallet analysis summary
                wallet_counts = {
                    "total_wallets": len(wallet_df),
                }

                # Calculate behavior classifications if buy_sell_ratio exists
                if "buy_sell_ratio" in wallet_df.columns:
                    wallet_counts["mainly_buyers"] = len(wallet_df[wallet_df["buy_sell_ratio"] > 2])
                    wallet_counts["mainly_sellers"] = len(wallet_df[wallet_df["buy_sell_ratio"] < 0.5])
                    wallet_counts["balanced_traders"] = len(
                        wallet_df[(wallet_df["buy_sell_ratio"] >= 0.5) & (wallet_df["buy_sell_ratio"] <= 2)])

                    # Alternatively, if buy_ratio and sell_ratio exist
                elif "buy_ratio" in wallet_df.columns and "sell_ratio" in wallet_df.columns:
                    wallet_counts["mainly_buyers"] = len(wallet_df[wallet_df["buy_ratio"] > 0.7])
                    wallet_counts["mainly_sellers"] = len(wallet_df[wallet_df["sell_ratio"] > 0.7])
                    wallet_counts["balanced_traders"] = len(
                        wallet_df[(wallet_df["buy_ratio"] <= 0.7) & (wallet_df["sell_ratio"] <= 0.7)])

                self.add_section(
                    "Wallet Analysis Summary",
                    wallet_counts,
                    "summary"
                )

            self.update_progress(40.0)

            # Add wallet classification section
            if not wallet_df.empty:
                # Classify wallets based on available data
                if "buy_sell_ratio" in wallet_df.columns:
                    wallet_df["classification"] = wallet_df["buy_sell_ratio"].apply(
                        lambda x: "Buyer" if x > 2 else ("Seller" if x < 0.5 else "Balanced")
                    )
                elif "buy_ratio" in wallet_df.columns and "sell_ratio" in wallet_df.columns:
                    wallet_df["classification"] = wallet_df.apply(
                        lambda row: "Buyer" if row["buy_ratio"] > 0.7 else (
                            "Seller" if row["sell_ratio"] > 0.7 else "Balanced"),
                        axis=1
                    )

                self.add_section(
                    "Wallet Classification",
                    wallet_df,
                    "table"
                )

                # Group by classification if available
                if "classification" in wallet_df.columns:
                    classification_counts = wallet_df["classification"].value_counts().reset_index()
                    classification_counts.columns = ["classification", "count"]

                    self.add_section(
                        "Wallet Classification Distribution",
                        classification_counts,
                        "chart"
                    )

            self.update_progress(70.0)

        # Process specific wallet transactions if available
        if "specific_wallet" in self.data_sources and "transactions" in self.data_sources:
            wallet_address = self.data_sources["specific_wallet"]
            transactions = self.data_sources["transactions"]

            # Filter transactions for the specific wallet
            if not isinstance(transactions, pd.DataFrame):
                transactions_df = pd.DataFrame(transactions)
            else:
                transactions_df = transactions

            if "wallet_address" in transactions_df.columns:
                wallet_txs = transactions_df[transactions_df["wallet_address"] == wallet_address]

                if not wallet_txs.empty:
                    # Add specific wallet transaction history
                    self.add_section(
                        f"Transaction History for {wallet_address}",
                        wallet_txs,
                        "table"
                    )

                    # Add transaction type distribution if transaction_type exists
                    if "transaction_type" in wallet_txs.columns:
                        tx_type_counts = wallet_txs["transaction_type"].value_counts().reset_index()
                        tx_type_counts.columns = ["transaction_type", "count"]

                        self.add_section(
                            "Transaction Type Distribution",
                            tx_type_counts,
                            "chart"
                        )

        self.update_progress(100.0)

        return self.to_dict()


class ComprehensiveReport(ReportGenerator):
    """
    Comprehensive report generator that combines all report types.
    Includes executive summary and detailed sections for each analysis type.
    """

    async def generate(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report.

        Returns:
            Generated report as a dictionary
        """
        if not self.data_sources:
            raise ValueError("No data sources provided for report generation")

        self.update_progress(5.0)

        # Create executive summary
        if "job_result" in self.data_sources:
            job_result = self.data_sources["job_result"]
            summary_data = {
                "pair_id": job_result.get("pair_id", "N/A"),
                "analysis_period": job_result.get("analysis_period", {})
            }

            # Add threshold analysis summary
            if "threshold_analysis" in job_result:
                threshold = job_result["threshold_analysis"]
                summary_data["threshold_analysis"] = {
                    "significant_transactions": threshold.get("significant_transactions", 0),
                    "whale_wallets": threshold.get("whale_wallets", 0),
                }

            # Add wave analysis summary
            if "wave_analysis" in job_result:
                wave = job_result["wave_analysis"]
                summary_data["wave_analysis"] = {
                    "buy_waves": wave.get("buy_waves", 0),
                    "sell_waves": wave.get("sell_waves", 0),
                    "largest_buy_wave": wave.get("largest_buy_wave", 0),
                    "largest_sell_wave": wave.get("largest_sell_wave", 0),
                }

            # Add ratio analysis summary
            if "ratio_analysis" in job_result:
                ratio = job_result["ratio_analysis"]
                summary_data["ratio_analysis"] = {
                    "wallets_analyzed": ratio.get("wallets_analyzed", 0),
                    "buyers": ratio.get("buyers", 0),
                    "sellers": ratio.get("sellers", 0),
                    "balanced_traders": ratio.get("balanced_traders", 0),
                }

            # Add pattern analysis summary
            if "pattern_analysis" in job_result:
                pattern = job_result["pattern_analysis"]
                summary_data["pattern_analysis"] = {
                    "total_patterns": pattern.get("total_patterns", 0),
                    "pattern_types": pattern.get("pattern_types", {})
                }

            self.add_section(
                "Executive Summary",
                summary_data,
                "summary"
            )

        self.update_progress(20.0)

        # Process transaction data
        if "threshold_analysis" in self.data_sources:
            threshold_data = self.data_sources["threshold_analysis"]

            if isinstance(threshold_data, dict) and "significant_transactions" in threshold_data:
                # Create transaction report generator
                transaction_report = TransactionReport(self.session)
                await transaction_report.set_data_source("significant_transactions",
                                                         threshold_data["significant_transactions"])

                # Generate transaction report
                tx_report = await transaction_report.generate()

                # Include transaction report sections
                for section in tx_report.get("sections", []):
                    self.add_section(
                        f"Transactions - {section['title']}",
                        section["content"],
                        section["type"]
                    )

        self.update_progress(40.0)

        # Process wave data
        if "wave_analysis" in self.data_sources:
            wave_data = self.data_sources["wave_analysis"]

            if isinstance(wave_data, dict):
                # Create wave report generator
                wave_report = WaveReport(self.session)

                if "buy_waves" in wave_data:
                    await wave_report.set_data_source("buy_waves", wave_data["buy_waves"])

                if "sell_waves" in wave_data:
                    await wave_report.set_data_source("sell_waves", wave_data["sell_waves"])

                # Generate wave report
                wave_report_data = await wave_report.generate()

                # Include wave report sections
                for section in wave_report_data.get("sections", []):
                    self.add_section(
                        f"Waves - {section['title']}",
                        section["content"],
                        section["type"]
                    )

        self.update_progress(60.0)

        # Process wallet analysis data
        if "ratio_analysis" in self.data_sources:
            ratio_data = self.data_sources["ratio_analysis"]

            if isinstance(ratio_data, dict):
                # Create wallet analysis report generator
                wallet_report = WalletAnalysisReport(self.session)

                # Combine all wallet types into one dataset
                wallet_data = []

                for key in ["all_buys", "all_sells", "mostly_buys", "mostly_sells", "balanced"]:
                    if key in ratio_data:
                        wallets = ratio_data[key]
                        for wallet in wallets:
                            wallet["classification"] = key.replace("_", " ").title()
                            wallet_data.append(wallet)

                await wallet_report.set_data_source("wallet_analysis", wallet_data)

                # Generate wallet report
                wallet_report_data = await wallet_report.generate()

                # Include wallet report sections
                for section in wallet_report_data.get("sections", []):
                    self.add_section(
                        f"Wallets - {section['title']}",
                        section["content"],
                        section["type"]
                    )

        self.update_progress(80.0)

        # Process pattern data
        if "pattern_analysis" in self.data_sources:
            pattern_data = self.data_sources["pattern_analysis"]

            if isinstance(pattern_data, dict) and "patterns" in pattern_data:
                patterns = pattern_data["patterns"]

                # Convert to DataFrame
                patterns_df = pd.DataFrame(patterns) if not isinstance(patterns, pd.DataFrame) else patterns

                if not patterns_df.empty:
                    # Add patterns section
                    self.add_section(
                        "Detected Patterns",
                        patterns_df,
                        "table"
                    )

                    # Group by pattern type if available
                    if "type" in patterns_df.columns:
                        pattern_counts = patterns_df["type"].value_counts().reset_index()
                        pattern_counts.columns = ["pattern_type", "count"]

                        self.add_section(
                            "Pattern Type Distribution",
                            pattern_counts,
                            "chart"
                        )

        self.update_progress(100.0)

        return self.to_dict()

