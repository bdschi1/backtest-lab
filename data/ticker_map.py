"""US equity and ETF ticker-to-name mapping.

Provides bidirectional lookup between ticker symbols and company/fund names
for a broad US equity universe. Covers:
    - All ~500 S&P 500 constituents
    - Notable non-index US-listed stocks (large tech, ADRs, crypto-adjacent, etc.)
    - Major US-listed ETFs

Used for:
    - User-friendly display in tearsheets and UI
    - Ticker validation before data fetch
    - Theme/sector grouping (via GICS sector or "ETF" tag)

Known issues:
    - LLY: Eli Lilly & Co. In some data providers (particularly outside
      yfinance), the ticker LLY may resolve to a different security.
      yfinance correctly maps LLY -> Eli Lilly. If using Bloomberg or
      IB, verify the resolved instrument.

This mapping is a point-in-time snapshot. Index constituents change
quarterly (additions/deletions). This serves as a starting reference.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# US equity universe (S&P 500 + notable non-index + major ETFs)
# Updated: Feb 2026
# ---------------------------------------------------------------------------

US_EQUITY_MAP: dict[str, dict[str, str]] = {
    # Each entry: ticker -> {"name": company_name, "sector": GICS_sector}
    # For ETFs: sector = "ETF"
    # For ADRs: sector = the company's actual GICS sector

    # ===================================================================
    # S&P 500 constituents (~503 tickers including dual-class shares)
    # ===================================================================

    # --- A ---
    "A": {"name": "Agilent Technologies Inc.", "sector": "Health Care"},
    "AAPL": {"name": "Apple Inc.", "sector": "Information Technology"},
    "ABBV": {"name": "AbbVie Inc.", "sector": "Health Care"},
    "ABNB": {"name": "Airbnb Inc.", "sector": "Consumer Discretionary"},
    "ABT": {"name": "Abbott Laboratories", "sector": "Health Care"},
    "ACGL": {"name": "Arch Capital Group Ltd.", "sector": "Financials"},
    "ACN": {"name": "Accenture plc", "sector": "Information Technology"},
    "ADBE": {"name": "Adobe Inc.", "sector": "Information Technology"},
    "ADI": {"name": "Analog Devices Inc.", "sector": "Information Technology"},
    "ADM": {"name": "Archer-Daniels-Midland Co.", "sector": "Consumer Staples"},
    "ADP": {"name": "Automatic Data Processing Inc.", "sector": "Industrials"},
    "ADSK": {"name": "Autodesk Inc.", "sector": "Information Technology"},
    "AEE": {"name": "Ameren Corp.", "sector": "Utilities"},
    "AEP": {"name": "American Electric Power Co. Inc.", "sector": "Utilities"},
    "AES": {"name": "The AES Corp.", "sector": "Utilities"},
    "AFL": {"name": "Aflac Inc.", "sector": "Financials"},
    "AIG": {"name": "American International Group Inc.", "sector": "Financials"},
    "AIZ": {"name": "Assurant Inc.", "sector": "Financials"},
    "AJG": {"name": "Arthur J. Gallagher & Co.", "sector": "Financials"},
    "AKAM": {"name": "Akamai Technologies Inc.", "sector": "Information Technology"},
    "ALB": {"name": "Albemarle Corp.", "sector": "Materials"},
    "ALGN": {"name": "Align Technology Inc.", "sector": "Health Care"},
    "ALL": {"name": "The Allstate Corp.", "sector": "Financials"},
    "ALLE": {"name": "Allegion plc", "sector": "Industrials"},
    "AMAT": {"name": "Applied Materials Inc.", "sector": "Information Technology"},
    "AMCR": {"name": "Amcor plc", "sector": "Materials"},
    "AMD": {"name": "Advanced Micro Devices Inc.", "sector": "Information Technology"},
    "AME": {"name": "AMETEK Inc.", "sector": "Industrials"},
    "AMGN": {"name": "Amgen Inc.", "sector": "Health Care"},
    "AMP": {"name": "Ameriprise Financial Inc.", "sector": "Financials"},
    "AMT": {"name": "American Tower Corp.", "sector": "Real Estate"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
    "ANET": {"name": "Arista Networks Inc.", "sector": "Information Technology"},
    "ANSS": {"name": "ANSYS Inc.", "sector": "Information Technology"},
    "AON": {"name": "Aon plc", "sector": "Financials"},
    "AOS": {"name": "A. O. Smith Corp.", "sector": "Industrials"},
    "APA": {"name": "APA Corp.", "sector": "Energy"},
    "APD": {"name": "Air Products and Chemicals Inc.", "sector": "Materials"},
    "APH": {"name": "Amphenol Corp.", "sector": "Information Technology"},
    "APTV": {"name": "Aptiv PLC", "sector": "Consumer Discretionary"},
    "ARE": {"name": "Alexandria Real Estate Equities Inc.", "sector": "Real Estate"},
    "ATO": {"name": "Atmos Energy Corp.", "sector": "Utilities"},
    "AVGO": {"name": "Broadcom Inc.", "sector": "Information Technology"},
    "AVB": {"name": "AvalonBay Communities Inc.", "sector": "Real Estate"},
    "AVY": {"name": "Avery Dennison Corp.", "sector": "Materials"},
    "AWK": {"name": "American Water Works Co. Inc.", "sector": "Utilities"},
    "AXP": {"name": "American Express Co.", "sector": "Financials"},
    "AZO": {"name": "AutoZone Inc.", "sector": "Consumer Discretionary"},

    # --- B ---
    "BA": {"name": "The Boeing Co.", "sector": "Industrials"},
    "BAC": {"name": "Bank of America Corp.", "sector": "Financials"},
    "BALL": {"name": "Ball Corp.", "sector": "Materials"},
    "BAX": {"name": "Baxter International Inc.", "sector": "Health Care"},
    "BBWI": {"name": "Bath & Body Works Inc.", "sector": "Consumer Discretionary"},
    "BBY": {"name": "Best Buy Co. Inc.", "sector": "Consumer Discretionary"},
    "BDX": {"name": "Becton Dickinson and Co.", "sector": "Health Care"},
    "BEN": {"name": "Franklin Resources Inc.", "sector": "Financials"},
    "BF.B": {"name": "Brown-Forman Corp.", "sector": "Consumer Staples"},
    "BG": {"name": "Bunge Global SA", "sector": "Consumer Staples"},
    "BIIB": {"name": "Biogen Inc.", "sector": "Health Care"},
    "BIO": {"name": "Bio-Rad Laboratories Inc.", "sector": "Health Care"},
    "BK": {"name": "The Bank of New York Mellon Corp.", "sector": "Financials"},
    "BKNG": {"name": "Booking Holdings Inc.", "sector": "Consumer Discretionary"},
    "BKR": {"name": "Baker Hughes Co.", "sector": "Energy"},
    "BLK": {"name": "BlackRock Inc.", "sector": "Financials"},
    "BMY": {"name": "Bristol-Myers Squibb Co.", "sector": "Health Care"},
    "BR": {"name": "Broadridge Financial Solutions Inc.", "sector": "Industrials"},
    "BRK.B": {"name": "Berkshire Hathaway Inc.", "sector": "Financials"},
    "BRO": {"name": "Brown & Brown Inc.", "sector": "Financials"},
    "BSX": {"name": "Boston Scientific Corp.", "sector": "Health Care"},
    "BWA": {"name": "BorgWarner Inc.", "sector": "Consumer Discretionary"},

    # --- C ---
    "C": {"name": "Citigroup Inc.", "sector": "Financials"},
    "CAG": {"name": "Conagra Brands Inc.", "sector": "Consumer Staples"},
    "CAH": {"name": "Cardinal Health Inc.", "sector": "Health Care"},
    "CARR": {"name": "Carrier Global Corp.", "sector": "Industrials"},
    "CAT": {"name": "Caterpillar Inc.", "sector": "Industrials"},
    "CB": {"name": "Chubb Ltd.", "sector": "Financials"},
    "CBOE": {"name": "Cboe Global Markets Inc.", "sector": "Financials"},
    "CCI": {"name": "Crown Castle Inc.", "sector": "Real Estate"},
    "CCL": {"name": "Carnival Corp.", "sector": "Consumer Discretionary"},
    "CDNS": {"name": "Cadence Design Systems Inc.", "sector": "Information Technology"},
    "CDW": {"name": "CDW Corp.", "sector": "Information Technology"},
    "CE": {"name": "Celanese Corp.", "sector": "Materials"},
    "CEG": {"name": "Constellation Energy Corp.", "sector": "Utilities"},
    "CF": {"name": "CF Industries Holdings Inc.", "sector": "Materials"},
    "CFG": {"name": "Citizens Financial Group Inc.", "sector": "Financials"},
    "CHD": {"name": "Church & Dwight Co. Inc.", "sector": "Consumer Staples"},
    "CHRW": {"name": "C.H. Robinson Worldwide Inc.", "sector": "Industrials"},
    "CI": {"name": "The Cigna Group", "sector": "Health Care"},
    "CINF": {"name": "Cincinnati Financial Corp.", "sector": "Financials"},
    "CL": {"name": "Colgate-Palmolive Co.", "sector": "Consumer Staples"},
    "CLX": {"name": "The Clorox Co.", "sector": "Consumer Staples"},
    "CMA": {"name": "Comerica Inc.", "sector": "Financials"},
    "CME": {"name": "CME Group Inc.", "sector": "Financials"},
    "CMG": {"name": "Chipotle Mexican Grill Inc.", "sector": "Consumer Discretionary"},
    "CMI": {"name": "Cummins Inc.", "sector": "Industrials"},
    "CMCSA": {"name": "Comcast Corp.", "sector": "Communication Services"},
    "CMS": {"name": "CMS Energy Corp.", "sector": "Utilities"},
    "CNC": {"name": "Centene Corp.", "sector": "Health Care"},
    "CNP": {"name": "CenterPoint Energy Inc.", "sector": "Utilities"},
    "COF": {"name": "Capital One Financial Corp.", "sector": "Financials"},
    "COIN": {"name": "Coinbase Global Inc.", "sector": "Financials"},
    "COO": {"name": "The Cooper Companies Inc.", "sector": "Health Care"},
    "COP": {"name": "ConocoPhillips", "sector": "Energy"},
    "COR": {"name": "Cencora Inc.", "sector": "Health Care"},
    "COST": {"name": "Costco Wholesale Corp.", "sector": "Consumer Staples"},
    "CPB": {"name": "Campbell Soup Co.", "sector": "Consumer Staples"},
    "CPRT": {"name": "Copart Inc.", "sector": "Industrials"},
    "CPT": {"name": "Camden Property Trust", "sector": "Real Estate"},
    "CRL": {"name": "Charles River Laboratories International Inc.", "sector": "Health Care"},
    "CRM": {"name": "Salesforce Inc.", "sector": "Information Technology"},
    "CRWD": {"name": "CrowdStrike Holdings Inc.", "sector": "Information Technology"},
    "CSCO": {"name": "Cisco Systems Inc.", "sector": "Information Technology"},
    "CSGP": {"name": "CoStar Group Inc.", "sector": "Real Estate"},
    "CSX": {"name": "CSX Corp.", "sector": "Industrials"},
    "CTAS": {"name": "Cintas Corp.", "sector": "Industrials"},
    "CTLT": {"name": "Catalent Inc.", "sector": "Health Care"},
    "CTSH": {"name": "Cognizant Technology Solutions Corp.", "sector": "Information Technology"},
    "CTVA": {"name": "Corteva Inc.", "sector": "Materials"},
    "CVS": {"name": "CVS Health Corp.", "sector": "Health Care"},
    "CVX": {"name": "Chevron Corp.", "sector": "Energy"},
    "CZR": {"name": "Caesars Entertainment Inc.", "sector": "Consumer Discretionary"},

    # --- D ---
    "D": {"name": "Dominion Energy Inc.", "sector": "Utilities"},
    "DAL": {"name": "Delta Air Lines Inc.", "sector": "Industrials"},
    "DAY": {"name": "Dayforce Inc.", "sector": "Information Technology"},
    "DD": {"name": "DuPont de Nemours Inc.", "sector": "Materials"},
    "DE": {"name": "Deere & Co.", "sector": "Industrials"},
    "DECK": {"name": "Deckers Outdoor Corp.", "sector": "Consumer Discretionary"},
    "DFS": {"name": "Discover Financial Services", "sector": "Financials"},
    "DG": {"name": "Dollar General Corp.", "sector": "Consumer Discretionary"},
    "DGX": {"name": "Quest Diagnostics Inc.", "sector": "Health Care"},
    "DHI": {"name": "D.R. Horton Inc.", "sector": "Consumer Discretionary"},
    "DHR": {"name": "Danaher Corp.", "sector": "Health Care"},
    "DIS": {"name": "The Walt Disney Co.", "sector": "Communication Services"},
    "DLTR": {"name": "Dollar Tree Inc.", "sector": "Consumer Discretionary"},
    "DOV": {"name": "Dover Corp.", "sector": "Industrials"},
    "DOW": {"name": "Dow Inc.", "sector": "Materials"},
    "DPZ": {"name": "Domino's Pizza Inc.", "sector": "Consumer Discretionary"},
    "DRI": {"name": "Darden Restaurants Inc.", "sector": "Consumer Discretionary"},
    "DTE": {"name": "DTE Energy Co.", "sector": "Utilities"},
    "DUK": {"name": "Duke Energy Corp.", "sector": "Utilities"},
    "DVA": {"name": "DaVita Inc.", "sector": "Health Care"},
    "DVN": {"name": "Devon Energy Corp.", "sector": "Energy"},
    "DXCM": {"name": "DexCom Inc.", "sector": "Health Care"},
    "DXC": {"name": "DXC Technology Co.", "sector": "Information Technology"},

    # --- E ---
    "EA": {"name": "Electronic Arts Inc.", "sector": "Communication Services"},
    "EBAY": {"name": "eBay Inc.", "sector": "Consumer Discretionary"},
    "ECL": {"name": "Ecolab Inc.", "sector": "Materials"},
    "ED": {"name": "Consolidated Edison Inc.", "sector": "Utilities"},
    "EFX": {"name": "Equifax Inc.", "sector": "Industrials"},
    "EIX": {"name": "Edison International", "sector": "Utilities"},
    "EL": {"name": "The Estee Lauder Companies Inc.", "sector": "Consumer Staples"},
    "ELV": {"name": "Elevance Health Inc.", "sector": "Health Care"},
    "EMN": {"name": "Eastman Chemical Co.", "sector": "Materials"},
    "EMR": {"name": "Emerson Electric Co.", "sector": "Industrials"},
    "ENPH": {"name": "Enphase Energy Inc.", "sector": "Information Technology"},
    "EOG": {"name": "EOG Resources Inc.", "sector": "Energy"},
    "EPAM": {"name": "EPAM Systems Inc.", "sector": "Information Technology"},
    "EQIX": {"name": "Equinix Inc.", "sector": "Real Estate"},
    "EQR": {"name": "Equity Residential", "sector": "Real Estate"},
    "EQT": {"name": "EQT Corp.", "sector": "Energy"},
    "ES": {"name": "Eversource Energy", "sector": "Utilities"},
    "ESS": {"name": "Essex Property Trust Inc.", "sector": "Real Estate"},
    "ETN": {"name": "Eaton Corp. plc", "sector": "Industrials"},
    "ETR": {"name": "Entergy Corp.", "sector": "Utilities"},
    "ETSY": {"name": "Etsy Inc.", "sector": "Consumer Discretionary"},
    "EVRG": {"name": "Evergy Inc.", "sector": "Utilities"},
    "EW": {"name": "Edwards Lifesciences Corp.", "sector": "Health Care"},
    "EXC": {"name": "Exelon Corp.", "sector": "Utilities"},
    "EXPD": {"name": "Expeditors International of Washington Inc.", "sector": "Industrials"},
    "EXPE": {"name": "Expedia Group Inc.", "sector": "Consumer Discretionary"},
    "EXR": {"name": "Extra Space Storage Inc.", "sector": "Real Estate"},

    # --- F ---
    "F": {"name": "Ford Motor Co.", "sector": "Consumer Discretionary"},
    "FANG": {"name": "Diamondback Energy Inc.", "sector": "Energy"},
    "FAST": {"name": "Fastenal Co.", "sector": "Industrials"},
    "FBHS": {"name": "Fortune Brands Home & Security Inc.", "sector": "Industrials"},
    "FCX": {"name": "Freeport-McMoRan Inc.", "sector": "Materials"},
    "FDS": {"name": "FactSet Research Systems Inc.", "sector": "Financials"},
    "FDX": {"name": "FedEx Corp.", "sector": "Industrials"},
    "FE": {"name": "FirstEnergy Corp.", "sector": "Utilities"},
    "FFIV": {"name": "F5 Inc.", "sector": "Information Technology"},
    "FI": {"name": "Fiserv Inc.", "sector": "Financials"},
    "FICO": {"name": "Fair Isaac Corp.", "sector": "Information Technology"},
    "FIS": {"name": "Fidelity National Information Services Inc.", "sector": "Financials"},
    "FITB": {"name": "Fifth Third Bancorp", "sector": "Financials"},
    "FLT": {"name": "Corpay Inc.", "sector": "Financials"},
    "FMC": {"name": "FMC Corp.", "sector": "Materials"},
    "FOX": {"name": "Fox Corp. (Class B)", "sector": "Communication Services"},
    "FOXA": {"name": "Fox Corp. (Class A)", "sector": "Communication Services"},
    "FSLR": {"name": "First Solar Inc.", "sector": "Information Technology"},
    "FTNT": {"name": "Fortinet Inc.", "sector": "Information Technology"},

    # --- G ---
    "GD": {"name": "General Dynamics Corp.", "sector": "Industrials"},
    "GDDY": {"name": "GoDaddy Inc.", "sector": "Information Technology"},
    "GE": {"name": "GE Aerospace", "sector": "Industrials"},
    "GEHC": {"name": "GE HealthCare Technologies Inc.", "sector": "Health Care"},
    "GEN": {"name": "Gen Digital Inc.", "sector": "Information Technology"},
    "GEV": {"name": "GE Vernova Inc.", "sector": "Industrials"},
    "GILD": {"name": "Gilead Sciences Inc.", "sector": "Health Care"},
    "GIS": {"name": "General Mills Inc.", "sector": "Consumer Staples"},
    "GL": {"name": "Globe Life Inc.", "sector": "Financials"},
    "GLW": {"name": "Corning Inc.", "sector": "Information Technology"},
    "GM": {"name": "General Motors Co.", "sector": "Consumer Discretionary"},
    "GNRC": {"name": "Generac Holdings Inc.", "sector": "Industrials"},
    "GOOG": {"name": "Alphabet Inc. (Class C)", "sector": "Communication Services"},
    "GOOGL": {"name": "Alphabet Inc. (Class A)", "sector": "Communication Services"},
    "GPC": {"name": "Genuine Parts Co.", "sector": "Consumer Discretionary"},
    "GPN": {"name": "Global Payments Inc.", "sector": "Financials"},
    "GRMN": {"name": "Garmin Ltd.", "sector": "Consumer Discretionary"},
    "GS": {"name": "The Goldman Sachs Group Inc.", "sector": "Financials"},
    "GWW": {"name": "W.W. Grainger Inc.", "sector": "Industrials"},

    # --- H ---
    "HAL": {"name": "Halliburton Co.", "sector": "Energy"},
    "HBAN": {"name": "Huntington Bancshares Inc.", "sector": "Financials"},
    "HCA": {"name": "HCA Healthcare Inc.", "sector": "Health Care"},
    "HD": {"name": "The Home Depot Inc.", "sector": "Consumer Discretionary"},
    "HES": {"name": "Hess Corp.", "sector": "Energy"},
    "HIG": {"name": "The Hartford Financial Services Group Inc.", "sector": "Financials"},
    "HII": {"name": "Huntington Ingalls Industries Inc.", "sector": "Industrials"},
    "HLT": {"name": "Hilton Worldwide Holdings Inc.", "sector": "Consumer Discretionary"},
    "HOLX": {"name": "Hologic Inc.", "sector": "Health Care"},
    "HON": {"name": "Honeywell International Inc.", "sector": "Industrials"},
    "HPE": {"name": "Hewlett Packard Enterprise Co.", "sector": "Information Technology"},
    "HPQ": {"name": "HP Inc.", "sector": "Information Technology"},
    "HRL": {"name": "Hormel Foods Corp.", "sector": "Consumer Staples"},
    "HSIC": {"name": "Henry Schein Inc.", "sector": "Health Care"},
    "HST": {"name": "Host Hotels & Resorts Inc.", "sector": "Real Estate"},
    "HSY": {"name": "The Hershey Co.", "sector": "Consumer Staples"},
    "HUBB": {"name": "Hubbell Inc.", "sector": "Industrials"},
    "HUM": {"name": "Humana Inc.", "sector": "Health Care"},
    "HWM": {"name": "Howmet Aerospace Inc.", "sector": "Industrials"},

    # --- I ---
    "IBM": {"name": "International Business Machines Corp.", "sector": "Information Technology"},
    "ICE": {"name": "Intercontinental Exchange Inc.", "sector": "Financials"},
    "IDXX": {"name": "IDEXX Laboratories Inc.", "sector": "Health Care"},
    "IEX": {"name": "IDEX Corp.", "sector": "Industrials"},
    "IFF": {"name": "International Flavors & Fragrances Inc.", "sector": "Materials"},
    "ILMN": {"name": "Illumina Inc.", "sector": "Health Care"},
    "INCY": {"name": "Incyte Corp.", "sector": "Health Care"},
    "INTC": {"name": "Intel Corp.", "sector": "Information Technology"},
    "INTU": {"name": "Intuit Inc.", "sector": "Information Technology"},
    "INVH": {"name": "Invitation Homes Inc.", "sector": "Real Estate"},
    "IP": {"name": "International Paper Co.", "sector": "Materials"},
    "IPG": {"name": "The Interpublic Group of Companies Inc.", "sector": "Communication Services"},
    "IQV": {"name": "IQVIA Holdings Inc.", "sector": "Health Care"},
    "IR": {"name": "Ingersoll Rand Inc.", "sector": "Industrials"},
    "IRM": {"name": "Iron Mountain Inc.", "sector": "Real Estate"},
    "ISRG": {"name": "Intuitive Surgical Inc.", "sector": "Health Care"},
    "IT": {"name": "Gartner Inc.", "sector": "Information Technology"},
    "ITW": {"name": "Illinois Tool Works Inc.", "sector": "Industrials"},
    "IVZ": {"name": "Invesco Ltd.", "sector": "Financials"},

    # --- J ---
    "J": {"name": "Jacobs Solutions Inc.", "sector": "Industrials"},
    "JBHT": {"name": "J.B. Hunt Transport Services Inc.", "sector": "Industrials"},
    "JCI": {"name": "Johnson Controls International plc", "sector": "Industrials"},
    "JKHY": {"name": "Jack Henry & Associates Inc.", "sector": "Financials"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Health Care"},
    "JNPR": {"name": "Juniper Networks Inc.", "sector": "Information Technology"},
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financials"},

    # --- K ---
    "K": {"name": "Kellanova", "sector": "Consumer Staples"},
    "KDP": {"name": "Keurig Dr Pepper Inc.", "sector": "Consumer Staples"},
    "KEY": {"name": "KeyCorp", "sector": "Financials"},
    "KEYS": {"name": "Keysight Technologies Inc.", "sector": "Information Technology"},
    "KHC": {"name": "The Kraft Heinz Co.", "sector": "Consumer Staples"},
    "KIM": {"name": "Kimco Realty Corp.", "sector": "Real Estate"},
    "KLAC": {"name": "KLA Corp.", "sector": "Information Technology"},
    "KMB": {"name": "Kimberly-Clark Corp.", "sector": "Consumer Staples"},
    "KMI": {"name": "Kinder Morgan Inc.", "sector": "Energy"},
    "KMX": {"name": "CarMax Inc.", "sector": "Consumer Discretionary"},
    "KO": {"name": "The Coca-Cola Co.", "sector": "Consumer Staples"},
    "KR": {"name": "The Kroger Co.", "sector": "Consumer Staples"},

    # --- L ---
    "L": {"name": "Loews Corp.", "sector": "Financials"},
    "LDOS": {"name": "Leidos Holdings Inc.", "sector": "Industrials"},
    "LEN": {"name": "Lennar Corp.", "sector": "Consumer Discretionary"},
    "LH": {"name": "Labcorp Holdings Inc.", "sector": "Health Care"},
    "LHX": {"name": "L3Harris Technologies Inc.", "sector": "Industrials"},
    "LIN": {"name": "Linde plc", "sector": "Materials"},
    "LKQ": {"name": "LKQ Corp.", "sector": "Consumer Discretionary"},
    "LLY": {"name": "Eli Lilly & Co.", "sector": "Health Care"},  # See module docstring: known cross-provider issue
    "LMT": {"name": "Lockheed Martin Corp.", "sector": "Industrials"},
    "LNT": {"name": "Alliant Energy Corp.", "sector": "Utilities"},
    "LOW": {"name": "Lowe's Companies Inc.", "sector": "Consumer Discretionary"},
    "LPLA": {"name": "LPL Financial Holdings Inc.", "sector": "Financials"},
    "LRCX": {"name": "Lam Research Corp.", "sector": "Information Technology"},
    "LULU": {"name": "Lululemon Athletica Inc.", "sector": "Consumer Discretionary"},
    "LUV": {"name": "Southwest Airlines Co.", "sector": "Industrials"},
    "LVS": {"name": "Las Vegas Sands Corp.", "sector": "Consumer Discretionary"},
    "LW": {"name": "Lamb Weston Holdings Inc.", "sector": "Consumer Staples"},
    "LYB": {"name": "LyondellBasell Industries N.V.", "sector": "Materials"},
    "LYV": {"name": "Live Nation Entertainment Inc.", "sector": "Communication Services"},

    # --- M ---
    "MA": {"name": "Mastercard Inc.", "sector": "Financials"},
    "MAA": {"name": "Mid-America Apartment Communities Inc.", "sector": "Real Estate"},
    "MAR": {"name": "Marriott International Inc.", "sector": "Consumer Discretionary"},
    "MAS": {"name": "Masco Corp.", "sector": "Industrials"},
    "MCD": {"name": "McDonald's Corp.", "sector": "Consumer Discretionary"},
    "MCHP": {"name": "Microchip Technology Inc.", "sector": "Information Technology"},
    "MCK": {"name": "McKesson Corp.", "sector": "Health Care"},
    "MCO": {"name": "Moody's Corp.", "sector": "Financials"},
    "MDLZ": {"name": "Mondelez International Inc.", "sector": "Consumer Staples"},
    "MDT": {"name": "Medtronic plc", "sector": "Health Care"},
    "MET": {"name": "MetLife Inc.", "sector": "Financials"},
    "META": {"name": "Meta Platforms Inc.", "sector": "Communication Services"},
    "MGM": {"name": "MGM Resorts International", "sector": "Consumer Discretionary"},
    "MKC": {"name": "McCormick & Co. Inc.", "sector": "Consumer Staples"},
    "MKTX": {"name": "MarketAxess Holdings Inc.", "sector": "Financials"},
    "MLM": {"name": "Martin Marietta Materials Inc.", "sector": "Materials"},
    "MMC": {"name": "Marsh & McLennan Companies Inc.", "sector": "Financials"},
    "MMM": {"name": "3M Co.", "sector": "Industrials"},
    "MNST": {"name": "Monster Beverage Corp.", "sector": "Consumer Staples"},
    "MO": {"name": "Altria Group Inc.", "sector": "Consumer Staples"},
    "MOH": {"name": "Molina Healthcare Inc.", "sector": "Health Care"},
    "MOS": {"name": "The Mosaic Co.", "sector": "Materials"},
    "MPC": {"name": "Marathon Petroleum Corp.", "sector": "Energy"},
    "MPWR": {"name": "Monolithic Power Systems Inc.", "sector": "Information Technology"},
    "MRK": {"name": "Merck & Co. Inc.", "sector": "Health Care"},
    "MRNA": {"name": "Moderna Inc.", "sector": "Health Care"},
    "MRO": {"name": "Marathon Oil Corp.", "sector": "Energy"},
    "MRVL": {"name": "Marvell Technology Inc.", "sector": "Information Technology"},
    "MS": {"name": "Morgan Stanley", "sector": "Financials"},
    "MSCI": {"name": "MSCI Inc.", "sector": "Financials"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Information Technology"},
    "MSI": {"name": "Motorola Solutions Inc.", "sector": "Information Technology"},
    "MTCH": {"name": "Match Group Inc.", "sector": "Communication Services"},
    "MTB": {"name": "M&T Bank Corp.", "sector": "Financials"},
    "MTD": {"name": "Mettler-Toledo International Inc.", "sector": "Health Care"},
    "MU": {"name": "Micron Technology Inc.", "sector": "Information Technology"},

    # --- N ---
    "NCLH": {"name": "Norwegian Cruise Line Holdings Ltd.", "sector": "Consumer Discretionary"},
    "NDAQ": {"name": "Nasdaq Inc.", "sector": "Financials"},
    "NDSN": {"name": "Nordson Corp.", "sector": "Industrials"},
    "NEE": {"name": "NextEra Energy Inc.", "sector": "Utilities"},
    "NEM": {"name": "Newmont Corp.", "sector": "Materials"},
    "NFLX": {"name": "Netflix Inc.", "sector": "Communication Services"},
    "NI": {"name": "NiSource Inc.", "sector": "Utilities"},
    "NKE": {"name": "Nike Inc.", "sector": "Consumer Discretionary"},
    "NOC": {"name": "Northrop Grumman Corp.", "sector": "Industrials"},
    "NOW": {"name": "ServiceNow Inc.", "sector": "Information Technology"},
    "NRG": {"name": "NRG Energy Inc.", "sector": "Utilities"},
    "NSC": {"name": "Norfolk Southern Corp.", "sector": "Industrials"},
    "NTAP": {"name": "NetApp Inc.", "sector": "Information Technology"},
    "NTRS": {"name": "Northern Trust Corp.", "sector": "Financials"},
    "NUE": {"name": "Nucor Corp.", "sector": "Materials"},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Information Technology"},
    "NVR": {"name": "NVR Inc.", "sector": "Consumer Discretionary"},
    "NWL": {"name": "Newell Brands Inc.", "sector": "Consumer Discretionary"},
    "NWS": {"name": "News Corp. (Class B)", "sector": "Communication Services"},
    "NWSA": {"name": "News Corp. (Class A)", "sector": "Communication Services"},
    "NXPI": {"name": "NXP Semiconductors N.V.", "sector": "Information Technology"},
    "NXST": {"name": "Nexstar Media Group Inc.", "sector": "Communication Services"},

    # --- O ---
    "O": {"name": "Realty Income Corp.", "sector": "Real Estate"},
    "ODFL": {"name": "Old Dominion Freight Line Inc.", "sector": "Industrials"},
    "OKE": {"name": "ONEOK Inc.", "sector": "Energy"},
    "OMC": {"name": "Omnicom Group Inc.", "sector": "Communication Services"},
    "ON": {"name": "ON Semiconductor Corp.", "sector": "Information Technology"},
    "ORCL": {"name": "Oracle Corp.", "sector": "Information Technology"},
    "ORLY": {"name": "O'Reilly Automotive Inc.", "sector": "Consumer Discretionary"},
    "OTIS": {"name": "Otis Worldwide Corp.", "sector": "Industrials"},
    "OXY": {"name": "Occidental Petroleum Corp.", "sector": "Energy"},

    # --- P ---
    "PANW": {"name": "Palo Alto Networks Inc.", "sector": "Information Technology"},
    "PARA": {"name": "Paramount Global", "sector": "Communication Services"},
    "PAYC": {"name": "Paycom Software Inc.", "sector": "Information Technology"},
    "PAYX": {"name": "Paychex Inc.", "sector": "Industrials"},
    "PCAR": {"name": "PACCAR Inc.", "sector": "Industrials"},
    "PCG": {"name": "PG&E Corp.", "sector": "Utilities"},
    "PEAK": {"name": "Healthpeak Properties Inc.", "sector": "Real Estate"},
    "PEG": {"name": "Public Service Enterprise Group Inc.", "sector": "Utilities"},
    "PEP": {"name": "PepsiCo Inc.", "sector": "Consumer Staples"},
    "PFE": {"name": "Pfizer Inc.", "sector": "Health Care"},
    "PFG": {"name": "Principal Financial Group Inc.", "sector": "Financials"},
    "PG": {"name": "Procter & Gamble Co.", "sector": "Consumer Staples"},
    "PGR": {"name": "The Progressive Corp.", "sector": "Financials"},
    "PH": {"name": "Parker-Hannifin Corp.", "sector": "Industrials"},
    "PHM": {"name": "PulteGroup Inc.", "sector": "Consumer Discretionary"},
    "PKG": {"name": "Packaging Corp. of America", "sector": "Materials"},
    "PLD": {"name": "Prologis Inc.", "sector": "Real Estate"},
    "PM": {"name": "Philip Morris International Inc.", "sector": "Consumer Staples"},
    "PNC": {"name": "The PNC Financial Services Group Inc.", "sector": "Financials"},
    "PNR": {"name": "Pentair plc", "sector": "Industrials"},
    "PNW": {"name": "Pinnacle West Capital Corp.", "sector": "Utilities"},
    "PODD": {"name": "Insulet Corp.", "sector": "Health Care"},
    "POOL": {"name": "Pool Corp.", "sector": "Consumer Discretionary"},
    "PPG": {"name": "PPG Industries Inc.", "sector": "Materials"},
    "PPL": {"name": "PPL Corp.", "sector": "Utilities"},
    "PRU": {"name": "Prudential Financial Inc.", "sector": "Financials"},
    "PSA": {"name": "Public Storage", "sector": "Real Estate"},
    "PSX": {"name": "Phillips 66", "sector": "Energy"},
    "PTC": {"name": "PTC Inc.", "sector": "Information Technology"},
    "PVH": {"name": "PVH Corp.", "sector": "Consumer Discretionary"},
    "PWR": {"name": "Quanta Services Inc.", "sector": "Industrials"},
    "PYPL": {"name": "PayPal Holdings Inc.", "sector": "Financials"},

    # --- Q ---
    "QCOM": {"name": "QUALCOMM Inc.", "sector": "Information Technology"},
    "QRVO": {"name": "Qorvo Inc.", "sector": "Information Technology"},

    # --- R ---
    "RCL": {"name": "Royal Caribbean Cruises Ltd.", "sector": "Consumer Discretionary"},
    "RE": {"name": "Everest Group Ltd.", "sector": "Financials"},
    "REG": {"name": "Regency Centers Corp.", "sector": "Real Estate"},
    "REGN": {"name": "Regeneron Pharmaceuticals Inc.", "sector": "Health Care"},
    "RF": {"name": "Regions Financial Corp.", "sector": "Financials"},
    "RHI": {"name": "Robert Half Inc.", "sector": "Industrials"},
    "RJF": {"name": "Raymond James Financial Inc.", "sector": "Financials"},
    "RL": {"name": "Ralph Lauren Corp.", "sector": "Consumer Discretionary"},
    "RMD": {"name": "ResMed Inc.", "sector": "Health Care"},
    "ROK": {"name": "Rockwell Automation Inc.", "sector": "Industrials"},
    "ROL": {"name": "Rollins Inc.", "sector": "Industrials"},
    "ROP": {"name": "Roper Technologies Inc.", "sector": "Industrials"},
    "ROST": {"name": "Ross Stores Inc.", "sector": "Consumer Discretionary"},
    "RSG": {"name": "Republic Services Inc.", "sector": "Industrials"},
    "RTX": {"name": "RTX Corp.", "sector": "Industrials"},
    "RVTY": {"name": "Revvity Inc.", "sector": "Health Care"},

    # --- S ---
    "SBAC": {"name": "SBA Communications Corp.", "sector": "Real Estate"},
    "SBUX": {"name": "Starbucks Corp.", "sector": "Consumer Discretionary"},
    "SCHW": {"name": "The Charles Schwab Corp.", "sector": "Financials"},
    "SEDG": {"name": "SolarEdge Technologies Inc.", "sector": "Information Technology"},
    "SHW": {"name": "The Sherwin-Williams Co.", "sector": "Materials"},
    "SJM": {"name": "The J.M. Smucker Co.", "sector": "Consumer Staples"},
    "SLB": {"name": "Schlumberger Ltd.", "sector": "Energy"},
    "SMCI": {"name": "Super Micro Computer Inc.", "sector": "Information Technology"},
    "SNA": {"name": "Snap-on Inc.", "sector": "Industrials"},
    "SNPS": {"name": "Synopsys Inc.", "sector": "Information Technology"},
    "SO": {"name": "The Southern Co.", "sector": "Utilities"},
    "SOLV": {"name": "Solventum Corp.", "sector": "Health Care"},
    "SPG": {"name": "Simon Property Group Inc.", "sector": "Real Estate"},
    "SPGI": {"name": "S&P Global Inc.", "sector": "Financials"},
    "SRE": {"name": "Sempra", "sector": "Utilities"},
    "STE": {"name": "STERIS plc", "sector": "Health Care"},
    "STLD": {"name": "Steel Dynamics Inc.", "sector": "Materials"},
    "STT": {"name": "State Street Corp.", "sector": "Financials"},
    "STX": {"name": "Seagate Technology Holdings plc", "sector": "Information Technology"},
    "STZ": {"name": "Constellation Brands Inc.", "sector": "Consumer Staples"},
    "SWK": {"name": "Stanley Black & Decker Inc.", "sector": "Industrials"},
    "SWKS": {"name": "Skyworks Solutions Inc.", "sector": "Information Technology"},
    "SYF": {"name": "Synchrony Financial", "sector": "Financials"},
    "SYK": {"name": "Stryker Corp.", "sector": "Health Care"},
    "SYY": {"name": "Sysco Corp.", "sector": "Consumer Staples"},
    "SNOW": {"name": "Snowflake Inc.", "sector": "Information Technology"},

    # --- T ---
    "T": {"name": "AT&T Inc.", "sector": "Communication Services"},
    "TAP": {"name": "Molson Coors Beverage Co.", "sector": "Consumer Staples"},
    "TDG": {"name": "TransDigm Group Inc.", "sector": "Industrials"},
    "TDY": {"name": "Teledyne Technologies Inc.", "sector": "Industrials"},
    "TECH": {"name": "Bio-Techne Corp.", "sector": "Health Care"},
    "TEL": {"name": "TE Connectivity Ltd.", "sector": "Information Technology"},
    "TER": {"name": "Teradyne Inc.", "sector": "Information Technology"},
    "TFC": {"name": "Truist Financial Corp.", "sector": "Financials"},
    "TFX": {"name": "Teleflex Inc.", "sector": "Health Care"},
    "TGT": {"name": "Target Corp.", "sector": "Consumer Discretionary"},
    "TJX": {"name": "The TJX Companies Inc.", "sector": "Consumer Discretionary"},
    "TMO": {"name": "Thermo Fisher Scientific Inc.", "sector": "Health Care"},
    "TMUS": {"name": "T-Mobile US Inc.", "sector": "Communication Services"},
    "TPR": {"name": "Tapestry Inc.", "sector": "Consumer Discretionary"},
    "TRGP": {"name": "Targa Resources Corp.", "sector": "Energy"},
    "TRMB": {"name": "Trimble Inc.", "sector": "Information Technology"},
    "TROW": {"name": "T. Rowe Price Group Inc.", "sector": "Financials"},
    "TRV": {"name": "The Travelers Companies Inc.", "sector": "Financials"},
    "TSCO": {"name": "Tractor Supply Co.", "sector": "Consumer Discretionary"},
    "TSLA": {"name": "Tesla Inc.", "sector": "Consumer Discretionary"},
    "TSN": {"name": "Tyson Foods Inc.", "sector": "Consumer Staples"},
    "TT": {"name": "Trane Technologies plc", "sector": "Industrials"},
    "TTWO": {"name": "Take-Two Interactive Software Inc.", "sector": "Communication Services"},
    "TXN": {"name": "Texas Instruments Inc.", "sector": "Information Technology"},
    "TXT": {"name": "Textron Inc.", "sector": "Industrials"},
    "TYL": {"name": "Tyler Technologies Inc.", "sector": "Information Technology"},

    # --- U ---
    "UDR": {"name": "UDR Inc.", "sector": "Real Estate"},
    "ULTA": {"name": "Ulta Beauty Inc.", "sector": "Consumer Discretionary"},
    "UNH": {"name": "UnitedHealth Group Inc.", "sector": "Health Care"},
    "UNP": {"name": "Union Pacific Corp.", "sector": "Industrials"},
    "UPS": {"name": "United Parcel Service Inc.", "sector": "Industrials"},
    "URI": {"name": "United Rentals Inc.", "sector": "Industrials"},
    "USB": {"name": "U.S. Bancorp", "sector": "Financials"},

    # --- V ---
    "V": {"name": "Visa Inc.", "sector": "Financials"},
    "VICI": {"name": "VICI Properties Inc.", "sector": "Real Estate"},
    "VLO": {"name": "Valero Energy Corp.", "sector": "Energy"},
    "VMC": {"name": "Vulcan Materials Co.", "sector": "Materials"},
    "VRSK": {"name": "Verisk Analytics Inc.", "sector": "Financials"},
    "VRSN": {"name": "VeriSign Inc.", "sector": "Information Technology"},
    "VRTX": {"name": "Vertex Pharmaceuticals Inc.", "sector": "Health Care"},
    "VST": {"name": "Vistra Corp.", "sector": "Utilities"},
    "VTR": {"name": "Ventas Inc.", "sector": "Real Estate"},
    "VTRS": {"name": "Viatris Inc.", "sector": "Health Care"},
    "VZ": {"name": "Verizon Communications Inc.", "sector": "Communication Services"},

    # --- W ---
    "WAB": {"name": "Westinghouse Air Brake Technologies Corp.", "sector": "Industrials"},
    "WAT": {"name": "Waters Corp.", "sector": "Health Care"},
    "WBA": {"name": "Walgreens Boots Alliance Inc.", "sector": "Consumer Staples"},
    "WBD": {"name": "Warner Bros. Discovery Inc.", "sector": "Communication Services"},
    "WDC": {"name": "Western Digital Corp.", "sector": "Information Technology"},
    "WEC": {"name": "WEC Energy Group Inc.", "sector": "Utilities"},
    "WELL": {"name": "Welltower Inc.", "sector": "Real Estate"},
    "WFC": {"name": "Wells Fargo & Co.", "sector": "Financials"},
    "WM": {"name": "Waste Management Inc.", "sector": "Industrials"},
    "WMB": {"name": "The Williams Companies Inc.", "sector": "Energy"},
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer Staples"},
    "WRB": {"name": "W.R. Berkley Corp.", "sector": "Financials"},
    "WRK": {"name": "WestRock Co.", "sector": "Materials"},
    "WST": {"name": "West Pharmaceutical Services Inc.", "sector": "Health Care"},
    "WTW": {"name": "Willis Towers Watson plc", "sector": "Financials"},
    "WY": {"name": "Weyerhaeuser Co.", "sector": "Real Estate"},
    "WYNN": {"name": "Wynn Resorts Ltd.", "sector": "Consumer Discretionary"},

    # --- X ---
    "XEL": {"name": "Xcel Energy Inc.", "sector": "Utilities"},
    "XOM": {"name": "Exxon Mobil Corp.", "sector": "Energy"},
    "XYL": {"name": "Xylem Inc.", "sector": "Industrials"},

    # --- Y ---
    "YUM": {"name": "Yum! Brands Inc.", "sector": "Consumer Discretionary"},

    # --- Z ---
    "ZBH": {"name": "Zimmer Biomet Holdings Inc.", "sector": "Health Care"},
    "ZBRA": {"name": "Zebra Technologies Corp.", "sector": "Information Technology"},
    "ZION": {"name": "Zions Bancorporation N.A.", "sector": "Financials"},
    "ZTS": {"name": "Zoetis Inc.", "sector": "Health Care"},

    # ===================================================================
    # Notable non-S&P 500 US-listed stocks
    # ===================================================================

    # --- Large non-index tech ---
    "PLTR": {"name": "Palantir Technologies Inc.", "sector": "Information Technology"},
    "UBER": {"name": "Uber Technologies Inc.", "sector": "Industrials"},
    "DASH": {"name": "DoorDash Inc.", "sector": "Consumer Discretionary"},
    "RBLX": {"name": "Roblox Corp.", "sector": "Communication Services"},
    "U": {"name": "Unity Software Inc.", "sector": "Information Technology"},
    "SQ": {"name": "Block Inc.", "sector": "Financials"},
    "AFRM": {"name": "Affirm Holdings Inc.", "sector": "Financials"},
    "HOOD": {"name": "Robinhood Markets Inc.", "sector": "Financials"},
    "SOFI": {"name": "SoFi Technologies Inc.", "sector": "Financials"},
    "SNAP": {"name": "Snap Inc.", "sector": "Communication Services"},
    "PINS": {"name": "Pinterest Inc.", "sector": "Communication Services"},
    "NET": {"name": "Cloudflare Inc.", "sector": "Information Technology"},
    "DDOG": {"name": "Datadog Inc.", "sector": "Information Technology"},
    "ZS": {"name": "Zscaler Inc.", "sector": "Information Technology"},
    "MDB": {"name": "MongoDB Inc.", "sector": "Information Technology"},
    "TWLO": {"name": "Twilio Inc.", "sector": "Information Technology"},
    "OKTA": {"name": "Okta Inc.", "sector": "Information Technology"},

    # --- Major ADRs ---
    "BABA": {"name": "Alibaba Group Holding Ltd.", "sector": "Consumer Discretionary"},
    "TSM": {"name": "Taiwan Semiconductor Manufacturing Co. Ltd.", "sector": "Information Technology"},
    "NVO": {"name": "Novo Nordisk A/S", "sector": "Health Care"},
    "ASML": {"name": "ASML Holding N.V.", "sector": "Information Technology"},
    "SAP": {"name": "SAP SE", "sector": "Information Technology"},
    "TM": {"name": "Toyota Motor Corp.", "sector": "Consumer Discretionary"},
    "SONY": {"name": "Sony Group Corp.", "sector": "Consumer Discretionary"},
    "MELI": {"name": "MercadoLibre Inc.", "sector": "Consumer Discretionary"},
    "SE": {"name": "Sea Ltd.", "sector": "Communication Services"},
    "NU": {"name": "Nu Holdings Ltd.", "sector": "Financials"},
    "GRAB": {"name": "Grab Holdings Ltd.", "sector": "Consumer Discretionary"},
    "PDD": {"name": "PDD Holdings Inc.", "sector": "Consumer Discretionary"},
    "JD": {"name": "JD.com Inc.", "sector": "Consumer Discretionary"},
    "BIDU": {"name": "Baidu Inc.", "sector": "Communication Services"},

    # --- Crypto-adjacent ---
    "MSTR": {"name": "MicroStrategy Inc.", "sector": "Information Technology"},
    "MARA": {"name": "Marathon Digital Holdings Inc.", "sector": "Information Technology"},
    "RIOT": {"name": "Riot Platforms Inc.", "sector": "Information Technology"},

    # --- Recent high-profile IPOs ---
    "ARM": {"name": "Arm Holdings plc", "sector": "Information Technology"},
    "BIRK": {"name": "Birkenstock Holding plc", "sector": "Consumer Discretionary"},

    # --- Well-known mid-caps ---
    "RIVN": {"name": "Rivian Automotive Inc.", "sector": "Consumer Discretionary"},
    "LCID": {"name": "Lucid Group Inc.", "sector": "Consumer Discretionary"},
    "GME": {"name": "GameStop Corp.", "sector": "Consumer Discretionary"},
    "AMC": {"name": "AMC Entertainment Holdings Inc.", "sector": "Communication Services"},
    "CHWY": {"name": "Chewy Inc.", "sector": "Consumer Discretionary"},
    "W": {"name": "Wayfair Inc.", "sector": "Consumer Discretionary"},
    "CVNA": {"name": "Carvana Co.", "sector": "Consumer Discretionary"},
    "ROKU": {"name": "Roku Inc.", "sector": "Communication Services"},

    # ===================================================================
    # Major ETFs
    # ===================================================================
    "SPY": {"name": "SPDR S&P 500 ETF Trust", "sector": "ETF"},
    "QQQ": {"name": "Invesco QQQ Trust", "sector": "ETF"},
    "IWM": {"name": "iShares Russell 2000 ETF", "sector": "ETF"},
    "DIA": {"name": "SPDR Dow Jones Industrial Average ETF Trust", "sector": "ETF"},
    "XLF": {"name": "Financial Select Sector SPDR Fund", "sector": "ETF"},
    "XLK": {"name": "Technology Select Sector SPDR Fund", "sector": "ETF"},
    "XLE": {"name": "Energy Select Sector SPDR Fund", "sector": "ETF"},
    "XLV": {"name": "Health Care Select Sector SPDR Fund", "sector": "ETF"},
    "XLI": {"name": "Industrial Select Sector SPDR Fund", "sector": "ETF"},
    "XLP": {"name": "Consumer Staples Select Sector SPDR Fund", "sector": "ETF"},
    "XLU": {"name": "Utilities Select Sector SPDR Fund", "sector": "ETF"},
    "XLB": {"name": "Materials Select Sector SPDR Fund", "sector": "ETF"},
    "XLRE": {"name": "Real Estate Select Sector SPDR Fund", "sector": "ETF"},
    "XLC": {"name": "Communication Services Select Sector SPDR Fund", "sector": "ETF"},
    "XLY": {"name": "Consumer Discretionary Select Sector SPDR Fund", "sector": "ETF"},
    "GLD": {"name": "SPDR Gold Shares", "sector": "ETF"},
    "SLV": {"name": "iShares Silver Trust", "sector": "ETF"},
    "TLT": {"name": "iShares 20+ Year Treasury Bond ETF", "sector": "ETF"},
    "HYG": {"name": "iShares iBoxx $ High Yield Corporate Bond ETF", "sector": "ETF"},
    "LQD": {"name": "iShares iBoxx $ Investment Grade Corporate Bond ETF", "sector": "ETF"},
    "VXX": {"name": "iPath Series B S&P 500 VIX Short-Term Futures ETN", "sector": "ETF"},
    "EEM": {"name": "iShares MSCI Emerging Markets ETF", "sector": "ETF"},
    "EFA": {"name": "iShares MSCI EAFE ETF", "sector": "ETF"},
    "VTI": {"name": "Vanguard Total Stock Market ETF", "sector": "ETF"},
    "VOO": {"name": "Vanguard S&P 500 ETF", "sector": "ETF"},
    "BND": {"name": "Vanguard Total Bond Market ETF", "sector": "ETF"},
    "ARKK": {"name": "ARK Innovation ETF", "sector": "ETF"},
    "IBIT": {"name": "iShares Bitcoin Trust ETF", "sector": "ETF"},
}

# ---------------------------------------------------------------------------
# Known ticker issues
# ---------------------------------------------------------------------------

_KNOWN_TICKER_ISSUES: dict[str, str] = {
    "LLY": (
        "Eli Lilly & Co. — Some data providers outside yfinance may resolve "
        "LLY to a different security. yfinance correctly maps LLY to Eli Lilly. "
        "If using Bloomberg or IB, verify the resolved instrument matches "
        "Eli Lilly (NYSE: LLY, ISIN: US5324571083)."
    ),
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

def ticker_to_name(ticker: str) -> str | None:
    """Get company/fund name for a ticker. Returns None if not found."""
    entry = US_EQUITY_MAP.get(ticker.upper())
    return entry["name"] if entry else None


def ticker_to_sector(ticker: str) -> str | None:
    """Get GICS sector (or 'ETF') for a ticker. Returns None if not found."""
    entry = US_EQUITY_MAP.get(ticker.upper())
    return entry["sector"] if entry else None


def name_to_ticker(company_name: str) -> str | None:
    """Fuzzy reverse lookup: company/fund name -> ticker.

    Matches if the search string is contained in the company name
    (case-insensitive).
    """
    search = company_name.lower()
    for ticker, info in US_EQUITY_MAP.items():
        if search in info["name"].lower():
            return ticker
    return None


def get_sector_map(tickers: list[str]) -> dict[str, str]:
    """Build {ticker: sector} map for a list of tickers.

    Unknown tickers are mapped to "Unknown".
    """
    result = {}
    for t in tickers:
        sector = ticker_to_sector(t)
        result[t] = sector or "Unknown"
    return result


def validate_tickers(tickers: list[str]) -> list[str]:
    """Validate tickers and warn about known issues.

    Returns list of warning messages (empty if no issues).
    """
    warnings = []
    for t in tickers:
        t_upper = t.upper()
        if t_upper in _KNOWN_TICKER_ISSUES:
            warnings.append(f"\u26a0 {t_upper}: {_KNOWN_TICKER_ISSUES[t_upper]}")
        if t_upper not in US_EQUITY_MAP:
            # Not an error — just not in our equity map
            # Could be a valid ticker outside our coverage
            pass
    return warnings


def get_all_tickers() -> list[str]:
    """Return all tickers in the US equity map (S&P 500 + non-index + ETFs)."""
    return sorted(US_EQUITY_MAP.keys())


def get_all_sectors() -> list[str]:
    """Return all unique sectors (GICS sectors + 'ETF')."""
    sectors = set(info["sector"] for info in US_EQUITY_MAP.values())
    return sorted(sectors)
