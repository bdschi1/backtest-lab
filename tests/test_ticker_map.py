"""Tests for US equity ticker mapping."""

from data.ticker_map import (
    US_EQUITY_MAP,
    ticker_to_name,
    ticker_to_sector,
    name_to_ticker,
    get_sector_map,
    validate_tickers,
    get_all_tickers,
    get_all_sectors,
)


class TestTickerMap:

    def test_known_tickers(self):
        """Known S&P 500 tickers should resolve."""
        assert ticker_to_name("AAPL") == "Apple Inc."
        assert ticker_to_name("MSFT") == "Microsoft Corp."
        assert ticker_to_name("JPM") == "JPMorgan Chase & Co."

    def test_case_insensitive(self):
        """Lookups should be case-insensitive."""
        assert ticker_to_name("aapl") == "Apple Inc."
        assert ticker_to_sector("msft") == "Information Technology"

    def test_lly_is_eli_lilly(self):
        """LLY must map to Eli Lilly — known cross-provider issue."""
        assert ticker_to_name("LLY") == "Eli Lilly & Co."
        assert ticker_to_sector("LLY") == "Health Care"

    def test_unknown_ticker(self):
        """Unknown tickers return None."""
        assert ticker_to_name("ZZZZZ") is None
        assert ticker_to_sector("ZZZZZ") is None

    def test_sector_lookup(self):
        """Sector lookups should return GICS sectors."""
        assert ticker_to_sector("XOM") == "Energy"
        assert ticker_to_sector("NEE") == "Utilities"
        assert ticker_to_sector("AAPL") == "Information Technology"

    def test_reverse_lookup(self):
        """Company name to ticker should work."""
        assert name_to_ticker("Apple") == "AAPL"
        assert name_to_ticker("eli lilly") == "LLY"
        assert name_to_ticker("JPMorgan") == "JPM"

    def test_reverse_lookup_not_found(self):
        """Unknown company names return None."""
        assert name_to_ticker("Totally Fake Corp") is None

    def test_sector_map(self):
        """get_sector_map should build dict for ticker list."""
        result = get_sector_map(["AAPL", "JPM", "UNKNOWN"])
        assert result["AAPL"] == "Information Technology"
        assert result["JPM"] == "Financials"
        assert result["UNKNOWN"] == "Unknown"

    def test_validate_tickers_lly_warning(self):
        """LLY should trigger a warning."""
        warnings = validate_tickers(["AAPL", "LLY", "MSFT"])
        assert len(warnings) == 1
        assert "LLY" in warnings[0]
        assert "Eli Lilly" in warnings[0]

    def test_validate_tickers_no_warnings(self):
        """Normal tickers should produce no warnings."""
        warnings = validate_tickers(["AAPL", "MSFT", "GOOG"])
        assert len(warnings) == 0

    def test_all_tickers(self):
        """Should return sorted list of all tickers."""
        tickers = get_all_tickers()
        assert len(tickers) > 100
        assert tickers == sorted(tickers)

    def test_all_sectors(self):
        """Should return all GICS sectors."""
        sectors = get_all_sectors()
        assert "Information Technology" in sectors
        assert "Financials" in sectors
        assert "Health Care" in sectors
        assert "Energy" in sectors

    def test_map_has_etf_benchmarks(self):
        """Should include ETF benchmarks."""
        assert "SPY" in US_EQUITY_MAP
        assert "QQQ" in US_EQUITY_MAP

    def test_map_has_full_sp500_coverage(self):
        """Map should have 500+ entries (S&P 500 + non-index + ETFs)."""
        assert len(US_EQUITY_MAP) >= 500

    def test_non_index_stocks(self):
        """Should include notable non-S&P 500 stocks."""
        assert ticker_to_name("PLTR") == "Palantir Technologies Inc."
        assert ticker_to_name("RIVN") == "Rivian Automotive Inc."
        assert ticker_to_name("GME") == "GameStop Corp."

    def test_major_adrs(self):
        """Should include major ADRs."""
        assert ticker_to_name("BABA") == "Alibaba Group Holding Ltd."
        assert ticker_to_name("TSM") == "Taiwan Semiconductor Manufacturing Co. Ltd."
        assert ticker_to_sector("NVO") == "Health Care"

    def test_sector_etfs(self):
        """Sector ETFs should map with sector='ETF'."""
        assert ticker_to_sector("XLF") == "ETF"
        assert ticker_to_sector("XLE") == "ETF"
        assert ticker_to_sector("GLD") == "ETF"
        assert ticker_to_sector("TLT") == "ETF"

    def test_pfizer_example(self):
        """Pfizer should map correctly (user's example)."""
        assert ticker_to_name("PFE") == "Pfizer Inc."
        assert ticker_to_sector("PFE") == "Health Care"

    def test_etf_in_sectors(self):
        """'ETF' should appear in get_all_sectors."""
        sectors = get_all_sectors()
        assert "ETF" in sectors
