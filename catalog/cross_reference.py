"""
Catalog Cross-Reference System for Anomaly Verification

Queries astronomical databases (SIMBAD, NED, VizieR) to check if detected
anomalies are known objects or potentially new discoveries.

This helps distinguish:
- True discoveries: Objects not in any catalog
- Known objects: Already cataloged (false positive from our perspective)
- Model errors: Artifacts or processing issues
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "astrolens_artifacts"
CROSS_REF_LOG = ARTIFACTS_DIR / "data" / "cross_reference_log.json"
CROSS_REF_RESULTS = ARTIFACTS_DIR / "data" / "cross_reference_results.json"


@dataclass
class CatalogMatch:
    """A match found in an astronomical catalog."""
    catalog: str  # SIMBAD, NED, VizieR
    object_name: str
    object_type: str
    distance_arcsec: float
    ra: float
    dec: float
    bibcodes: List[str] = field(default_factory=list)  # Published papers
    url: str = ""
    raw_data: Dict = field(default_factory=dict)


@dataclass
class CrossReferenceResult:
    """Result of cross-referencing an anomaly against catalogs."""
    image_id: int
    image_path: str
    query_ra: float
    query_dec: float
    query_radius_arcsec: float
    
    # Status
    is_known: bool = False
    is_published: bool = False  # Has associated publications
    status: str = "unknown"  # known, unknown, error, artifact
    
    # Matches found
    matches: List[CatalogMatch] = field(default_factory=list)
    primary_match: Optional[CatalogMatch] = None
    
    # Metadata
    queried_at: str = ""
    query_duration_ms: int = 0
    error_message: str = ""
    
    # User verification (for training)
    human_verified: bool = False
    human_label: str = ""  # true_positive, false_positive, uncertain
    verified_by: str = ""
    verified_at: str = ""


class CatalogCrossReference:
    """
    Cross-reference astronomical images against major catalogs.
    
    Supported catalogs:
    - SIMBAD: The most comprehensive astronomical database
    - NED: NASA/IPAC Extragalactic Database (galaxies focus)
    - VizieR: Access to many published catalogs
    """
    
    def __init__(
        self,
        search_radius_arcsec: float = 60.0,  # 60 arcsec = 1 arcmin (appropriate for galaxies)
        timeout_seconds: float = 30.0,
    ):
        self.search_radius = search_radius_arcsec
        self.timeout = timeout_seconds
        self.results: Dict[int, CrossReferenceResult] = {}
        self._load_results()
    
    def _load_results(self):
        """Load previous cross-reference results."""
        if CROSS_REF_RESULTS.exists():
            try:
                with open(CROSS_REF_RESULTS, "r") as f:
                    data = json.load(f)
                    for item in data.get("results", []):
                        result = self._dict_to_result(item)
                        self.results[result.image_id] = result
                logger.info(f"Loaded {len(self.results)} previous cross-reference results")
            except Exception as e:
                logger.warning(f"Could not load cross-reference results: {e}")
    
    def _dict_to_result(self, data: Dict) -> CrossReferenceResult:
        """Convert dict to CrossReferenceResult."""
        matches = []
        for m in data.get("matches", []):
            matches.append(CatalogMatch(**m))
        
        primary = None
        if data.get("primary_match"):
            primary = CatalogMatch(**data["primary_match"])
        
        return CrossReferenceResult(
            image_id=data["image_id"],
            image_path=data["image_path"],
            query_ra=data["query_ra"],
            query_dec=data["query_dec"],
            query_radius_arcsec=data["query_radius_arcsec"],
            is_known=data.get("is_known", False),
            is_published=data.get("is_published", False),
            status=data.get("status", "unknown"),
            matches=matches,
            primary_match=primary,
            queried_at=data.get("queried_at", ""),
            query_duration_ms=data.get("query_duration_ms", 0),
            error_message=data.get("error_message", ""),
            human_verified=data.get("human_verified", False),
            human_label=data.get("human_label", ""),
            verified_by=data.get("verified_by", ""),
            verified_at=data.get("verified_at", ""),
        )
    
    def _save_results(self):
        """Save cross-reference results."""
        CROSS_REF_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        results_list = []
        for result in self.results.values():
            result_dict = asdict(result)
            results_list.append(result_dict)
        
        with open(CROSS_REF_RESULTS, "w") as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "total_results": len(results_list),
                "results": results_list,
            }, f, indent=2)
    
    def _log_query(
        self,
        image_id: int,
        catalog: str,
        ra: float,
        dec: float,
        success: bool,
        matches_found: int,
        duration_ms: int,
        error: str = "",
    ):
        """Append to query log for analysis."""
        CROSS_REF_LOG.parent.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "image_id": image_id,
            "catalog": catalog,
            "ra": ra,
            "dec": dec,
            "radius_arcsec": self.search_radius,
            "success": success,
            "matches_found": matches_found,
            "duration_ms": duration_ms,
            "error": error,
        }
        
        # Append to log file
        try:
            if CROSS_REF_LOG.exists():
                with open(CROSS_REF_LOG, "r") as f:
                    log_data = json.load(f)
            else:
                log_data = {"queries": []}
            
            log_data["queries"].append(log_entry)
            log_data["last_updated"] = datetime.now().isoformat()
            log_data["total_queries"] = len(log_data["queries"])
            
            with open(CROSS_REF_LOG, "w") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not update query log: {e}")
    
    def extract_coordinates(self, image_path: str) -> Optional[Tuple[float, float]]:
        """
        Extract RA/Dec coordinates from image filename or metadata.
        
        Supports patterns like:
        - gz_anomaly_0001_ra183.3_dec13.7.jpg
        - sdss_0001_ra200.6_dec40.6.jpg
        - Any filename with ra###.#_dec###.# pattern
        """
        filename = Path(image_path).name
        
        # Pattern: ra###.#_dec###.#
        pattern = r'ra([-+]?\d+\.?\d*)_dec([-+]?\d+\.?\d*)'
        match = re.search(pattern, filename, re.IGNORECASE)
        
        if match:
            ra = float(match.group(1))
            dec = float(match.group(2))
            return (ra, dec)
        
        # TODO: Add FITS header reading for images with WCS
        
        return None
    
    def query_simbad(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = None,
    ) -> List[CatalogMatch]:
        """
        Query SIMBAD astronomical database using TAP/ADQL.
        
        SIMBAD is the most comprehensive database of astronomical objects
        outside the Solar System.
        """
        if radius_arcsec is None:
            radius_arcsec = self.search_radius
        
        radius_deg = radius_arcsec / 3600.0
        
        # Use SIMBAD TAP service with ADQL query (more reliable than script)
        # Cone search using CONTAINS and CIRCLE
        adql_query = f"""
SELECT TOP 20
    main_id, otype_txt, ra, dec, nbref
FROM basic
WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
ORDER BY nbref DESC
"""
        
        tap_url = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"
        
        try:
            response = httpx.post(
                tap_url,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "json",
                    "query": adql_query,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            matches = []
            
            try:
                data = response.json()
                # SIMBAD TAP returns data in VOTable-like JSON format
                if "data" in data:
                    rows = data["data"]
                elif isinstance(data, list):
                    rows = data
                else:
                    rows = []
                
                for row in rows:
                    try:
                        if isinstance(row, dict):
                            obj_name = row.get("main_id", "Unknown")
                            obj_type = row.get("otype_txt", "Unknown")
                            obj_ra = float(row.get("ra", ra))
                            obj_dec = float(row.get("dec", dec))
                            nbref = int(row.get("nbref", 0))
                        else:
                            # Array format: [main_id, otype_txt, ra, dec, nbref]
                            obj_name = str(row[0]) if len(row) > 0 else "Unknown"
                            obj_type = str(row[1]) if len(row) > 1 else "Unknown"
                            obj_ra = float(row[2]) if len(row) > 2 else ra
                            obj_dec = float(row[3]) if len(row) > 3 else dec
                            nbref = int(row[4]) if len(row) > 4 else 0
                        
                        # Calculate angular separation
                        dist = self._angular_separation(ra, dec, obj_ra, obj_dec)
                        
                        matches.append(CatalogMatch(
                            catalog="SIMBAD",
                            object_name=obj_name,
                            object_type=obj_type,
                            distance_arcsec=dist * 3600,
                            ra=obj_ra,
                            dec=obj_dec,
                            bibcodes=[f"{nbref} refs"] if nbref > 0 else [],
                            url=f"https://simbad.u-strasbg.fr/simbad/sim-id?Ident={quote(obj_name)}",
                        ))
                    except (ValueError, IndexError, TypeError) as e:
                        logger.debug(f"Could not parse SIMBAD row: {row} - {e}")
                        continue
            except json.JSONDecodeError:
                # Try VOTable XML parsing as fallback
                logger.debug("SIMBAD returned non-JSON, trying VOTable parse")
                matches = self._parse_votable_simbad(response.text, ra, dec)
            
            return matches
            
        except Exception as e:
            logger.warning(f"SIMBAD query failed: {e}")
            return []
    
    def _parse_votable_simbad(self, xml_text: str, ra: float, dec: float) -> List[CatalogMatch]:
        """Parse SIMBAD VOTable XML response."""
        import xml.etree.ElementTree as ET
        
        matches = []
        try:
            root = ET.fromstring(xml_text)
            
            # Find TABLEDATA
            ns = {'vot': 'http://www.ivoa.net/xml/VOTable/v1.3'}
            
            for tr in root.findall('.//TR', ns) or root.findall('.//tr'):
                tds = tr.findall('TD', ns) or tr.findall('td') or list(tr)
                if len(tds) >= 4:
                    try:
                        obj_name = tds[0].text or "Unknown"
                        obj_type = tds[1].text or "Unknown" if len(tds) > 1 else "Unknown"
                        obj_ra = float(tds[2].text) if len(tds) > 2 and tds[2].text else ra
                        obj_dec = float(tds[3].text) if len(tds) > 3 and tds[3].text else dec
                        
                        dist = self._angular_separation(ra, dec, obj_ra, obj_dec)
                        
                        matches.append(CatalogMatch(
                            catalog="SIMBAD",
                            object_name=obj_name,
                            object_type=obj_type,
                            distance_arcsec=dist * 3600,
                            ra=obj_ra,
                            dec=obj_dec,
                            url=f"https://simbad.u-strasbg.fr/simbad/sim-id?Ident={quote(obj_name)}",
                        ))
                    except (ValueError, AttributeError):
                        continue
        except ET.ParseError:
            pass
        
        return matches
    
    def query_ned(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = None,
    ) -> List[CatalogMatch]:
        """
        Query NASA/IPAC Extragalactic Database (NED) using TAP service.
        
        NED specializes in extragalactic objects (galaxies, quasars, etc.)
        """
        if radius_arcsec is None:
            radius_arcsec = self.search_radius
        
        radius_deg = radius_arcsec / 3600.0
        
        # Use NED TAP service with ADQL
        adql_query = f"""
SELECT TOP 20
    prefname, pretype, ra, dec, z
FROM objdir
WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
"""
        
        tap_url = "https://ned.ipac.caltech.edu/tap/sync"
        
        try:
            response = httpx.post(
                tap_url,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "json",
                    "query": adql_query,
                },
                timeout=self.timeout,
            )
            
            if response.status_code != 200:
                # Fallback to simple cone search
                return self._query_ned_cone(ra, dec, radius_arcsec)
            
            matches = []
            
            try:
                data = response.json()
                rows = data.get("data", [])
                
                for row in rows:
                    try:
                        if isinstance(row, dict):
                            obj_name = row.get("prefname", "Unknown")
                            obj_type = row.get("pretype", "Unknown")
                            obj_ra = float(row.get("ra", ra))
                            obj_dec = float(row.get("dec", dec))
                        else:
                            obj_name = str(row[0]) if len(row) > 0 else "Unknown"
                            obj_type = str(row[1]) if len(row) > 1 else "Unknown"
                            obj_ra = float(row[2]) if len(row) > 2 else ra
                            obj_dec = float(row[3]) if len(row) > 3 else dec
                        
                        dist = self._angular_separation(ra, dec, obj_ra, obj_dec)
                        
                        matches.append(CatalogMatch(
                            catalog="NED",
                            object_name=obj_name,
                            object_type=obj_type,
                            distance_arcsec=dist * 3600,
                            ra=obj_ra,
                            dec=obj_dec,
                            url=f"https://ned.ipac.caltech.edu/byname?objname={quote(obj_name)}",
                        ))
                    except (ValueError, IndexError, TypeError):
                        continue
            except json.JSONDecodeError:
                return self._query_ned_cone(ra, dec, radius_arcsec)
            
            return matches
            
        except Exception as e:
            logger.warning(f"NED TAP query failed, trying cone search: {e}")
            return self._query_ned_cone(ra, dec, radius_arcsec)
    
    def _query_ned_cone(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
    ) -> List[CatalogMatch]:
        """Fallback NED cone search."""
        radius_arcmin = radius_arcsec / 60.0
        
        # NED simple cone search (IAU format)
        url = (
            f"https://ned.ipac.caltech.edu/cgi-bin/nph-objsearch"
            f"?search_type=Near+Position+Search"
            f"&in_csys=Equatorial"
            f"&in_equinox=J2000.0"
            f"&lon={ra:.6f}d"
            f"&lat={dec:.6f}d"
            f"&radius={radius_arcmin:.3f}"
            f"&out_csys=Equatorial"
            f"&out_equinox=J2000.0"
            f"&of=ascii_tab"
            f"&nmp_op=ANY"
        )
        
        try:
            response = httpx.get(url, timeout=self.timeout, follow_redirects=True)
            
            matches = []
            lines = response.text.strip().split("\n")
            
            # Skip header lines (start with #)
            data_started = False
            for line in lines:
                if line.startswith("#"):
                    continue
                if not data_started:
                    data_started = True
                    continue  # Skip column header
                
                parts = line.split("\t")
                if len(parts) >= 4:
                    try:
                        obj_name = parts[1].strip() if len(parts) > 1 else "Unknown"
                        obj_type = parts[4].strip() if len(parts) > 4 else "Unknown"
                        obj_ra = float(parts[2]) if len(parts) > 2 else ra
                        obj_dec = float(parts[3]) if len(parts) > 3 else dec
                        
                        dist = self._angular_separation(ra, dec, obj_ra, obj_dec)
                        
                        matches.append(CatalogMatch(
                            catalog="NED",
                            object_name=obj_name,
                            object_type=obj_type,
                            distance_arcsec=dist * 3600,
                            ra=obj_ra,
                            dec=obj_dec,
                            url=f"https://ned.ipac.caltech.edu/byname?objname={quote(obj_name)}",
                        ))
                    except (ValueError, IndexError):
                        continue
            
            return matches
            
        except Exception as e:
            logger.warning(f"NED cone search failed: {e}")
            return []
    
    def query_vizier(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = None,
    ) -> List[CatalogMatch]:
        """
        Query VizieR for Galaxy Zoo and other galaxy catalogs.
        
        Searches major galaxy catalogs: 2MASS XSC, SDSS galaxies, etc.
        """
        if radius_arcsec is None:
            radius_arcsec = self.search_radius
        
        radius_deg = radius_arcsec / 3600.0
        
        # VizieR TAP service - search multiple catalogs
        # VII/233 = 2MASS Extended Source Catalog
        # V/147 = SDSS DR12 PhotoObj
        adql_query = f"""
SELECT TOP 10
    _r, RAJ2000, DEJ2000, objID
FROM "V/147/sdss12"
WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
"""
        
        tap_url = "https://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync"
        
        try:
            response = httpx.post(
                tap_url,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "json",
                    "query": adql_query,
                },
                timeout=self.timeout,
            )
            
            if response.status_code != 200:
                return []
            
            matches = []
            
            try:
                data = response.json()
                rows = data.get("data", [])
                
                for row in rows:
                    try:
                        if isinstance(row, dict):
                            obj_ra = float(row.get("RAJ2000", ra))
                            obj_dec = float(row.get("DEJ2000", dec))
                            obj_id = str(row.get("objID", ""))
                        else:
                            obj_ra = float(row[1]) if len(row) > 1 else ra
                            obj_dec = float(row[2]) if len(row) > 2 else dec
                            obj_id = str(row[3]) if len(row) > 3 else ""
                        
                        dist = self._angular_separation(ra, dec, obj_ra, obj_dec)
                        
                        matches.append(CatalogMatch(
                            catalog="SDSS",
                            object_name=f"SDSS J{obj_id}" if obj_id else f"SDSS {obj_ra:.4f}{obj_dec:+.4f}",
                            object_type="Galaxy",
                            distance_arcsec=dist * 3600,
                            ra=obj_ra,
                            dec=obj_dec,
                            url=f"https://skyserver.sdss.org/dr18/VisualTools/explore/summary?objId={obj_id}" if obj_id else "",
                        ))
                    except (ValueError, IndexError, TypeError):
                        continue
            except json.JSONDecodeError:
                pass
            
            return matches
            
        except Exception as e:
            logger.debug(f"VizieR query failed: {e}")
            return []
    
    def _angular_separation(
        self,
        ra1: float,
        dec1: float,
        ra2: float,
        dec2: float,
    ) -> float:
        """Calculate angular separation in degrees."""
        import math
        
        # Convert to radians
        ra1_rad = math.radians(ra1)
        dec1_rad = math.radians(dec1)
        ra2_rad = math.radians(ra2)
        dec2_rad = math.radians(dec2)
        
        # Haversine formula
        dra = ra2_rad - ra1_rad
        ddec = dec2_rad - dec1_rad
        
        a = math.sin(ddec/2)**2 + math.cos(dec1_rad) * math.cos(dec2_rad) * math.sin(dra/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return math.degrees(c)
    
    def cross_reference(
        self,
        image_id: int,
        image_path: str,
        ra: float = None,
        dec: float = None,
        force: bool = False,
    ) -> CrossReferenceResult:
        """
        Cross-reference a single anomaly against catalogs.
        
        Args:
            image_id: Database ID of the image
            image_path: Path to the image file
            ra: RA coordinate (extracted from filename if not provided)
            dec: Dec coordinate (extracted from filename if not provided)
            force: Re-query even if already in results
        
        Returns:
            CrossReferenceResult with match information
        """
        # Check if already queried
        if not force and image_id in self.results:
            return self.results[image_id]
        
        # Extract coordinates if not provided
        if ra is None or dec is None:
            coords = self.extract_coordinates(image_path)
            if coords is None:
                return CrossReferenceResult(
                    image_id=image_id,
                    image_path=image_path,
                    query_ra=0,
                    query_dec=0,
                    query_radius_arcsec=self.search_radius,
                    status="error",
                    error_message="Could not extract coordinates from filename",
                    queried_at=datetime.now().isoformat(),
                )
            ra, dec = coords
        
        start_time = time.time()
        all_matches = []
        
        # Query SIMBAD
        try:
            simbad_matches = self.query_simbad(ra, dec)
            all_matches.extend(simbad_matches)
            self._log_query(
                image_id, "SIMBAD", ra, dec,
                success=True,
                matches_found=len(simbad_matches),
                duration_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            self._log_query(
                image_id, "SIMBAD", ra, dec,
                success=False,
                matches_found=0,
                duration_ms=int((time.time() - start_time) * 1000),
                error=str(e),
            )
        
        # Query NED
        try:
            ned_start = time.time()
            ned_matches = self.query_ned(ra, dec)
            all_matches.extend(ned_matches)
            self._log_query(
                image_id, "NED", ra, dec,
                success=True,
                matches_found=len(ned_matches),
                duration_ms=int((time.time() - ned_start) * 1000),
            )
        except Exception as e:
            self._log_query(
                image_id, "NED", ra, dec,
                success=False,
                matches_found=0,
                duration_ms=int((time.time() - time.time()) * 1000),
                error=str(e),
            )
        
        # Query VizieR (SDSS catalog)
        try:
            vizier_start = time.time()
            vizier_matches = self.query_vizier(ra, dec)
            all_matches.extend(vizier_matches)
            self._log_query(
                image_id, "VizieR", ra, dec,
                success=True,
                matches_found=len(vizier_matches),
                duration_ms=int((time.time() - vizier_start) * 1000),
            )
        except Exception as e:
            self._log_query(
                image_id, "VizieR", ra, dec,
                success=False,
                matches_found=0,
                duration_ms=int((time.time() - vizier_start) * 1000),
                error=str(e),
            )
        
        # Sort by distance
        all_matches.sort(key=lambda m: m.distance_arcsec)
        
        # Determine status
        is_known = len(all_matches) > 0
        is_published = any(m.bibcodes for m in all_matches)
        primary_match = all_matches[0] if all_matches else None
        
        if is_known:
            status = "known"
        else:
            status = "unknown"  # Potentially new discovery!
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        result = CrossReferenceResult(
            image_id=image_id,
            image_path=image_path,
            query_ra=ra,
            query_dec=dec,
            query_radius_arcsec=self.search_radius,
            is_known=is_known,
            is_published=is_published,
            status=status,
            matches=all_matches,
            primary_match=primary_match,
            queried_at=datetime.now().isoformat(),
            query_duration_ms=duration_ms,
        )
        
        # Store result
        self.results[image_id] = result
        self._save_results()
        
        return result
    
    def cross_reference_all(
        self,
        anomalies: List[Dict],
        progress_callback=None,
        delay_between_queries: float = 0.5,  # Be nice to catalog servers
    ) -> Dict[str, int]:
        """
        Cross-reference all anomalies against catalogs.
        
        Args:
            anomalies: List of dicts with 'id' and 'filepath' keys
            progress_callback: Optional callback(current, total, result)
            delay_between_queries: Delay between API calls (seconds)
        
        Returns:
            Summary statistics
        """
        stats = {
            "total": len(anomalies),
            "known": 0,
            "unknown": 0,
            "published": 0,
            "errors": 0,
            "skipped": 0,
        }
        
        for i, anomaly in enumerate(anomalies):
            image_id = anomaly.get("id") or anomaly.get("image_id")
            filepath = anomaly.get("filepath") or anomaly.get("image_path")
            
            if image_id is None or filepath is None:
                stats["errors"] += 1
                continue
            
            # Skip if already processed
            if image_id in self.results:
                result = self.results[image_id]
                stats["skipped"] += 1
            else:
                try:
                    result = self.cross_reference(image_id, filepath)
                    time.sleep(delay_between_queries)
                except Exception as e:
                    logger.warning(f"Error cross-referencing {image_id}: {e}")
                    stats["errors"] += 1
                    continue
            
            # Update stats
            if result.status == "known":
                stats["known"] += 1
            elif result.status == "unknown":
                stats["unknown"] += 1
            
            if result.is_published:
                stats["published"] += 1
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(anomalies), result)
        
        return stats
    
    def get_summary(self) -> Dict:
        """Get summary of cross-reference results."""
        known = sum(1 for r in self.results.values() if r.is_known)
        unknown = sum(1 for r in self.results.values() if not r.is_known)
        published = sum(1 for r in self.results.values() if r.is_published)
        verified = sum(1 for r in self.results.values() if r.human_verified)
        
        return {
            "total_checked": len(self.results),
            "known_objects": known,
            "unknown_objects": unknown,  # Potentially new!
            "with_publications": published,
            "human_verified": verified,
            "false_positives": sum(
                1 for r in self.results.values() 
                if r.human_label == "false_positive"
            ),
            "true_positives": sum(
                1 for r in self.results.values() 
                if r.human_label == "true_positive"
            ),
        }
    
    def mark_verified(
        self,
        image_id: int,
        label: str,  # true_positive, false_positive, uncertain
        verified_by: str = "user",
    ):
        """Mark a result as human-verified for training data."""
        if image_id in self.results:
            self.results[image_id].human_verified = True
            self.results[image_id].human_label = label
            self.results[image_id].verified_by = verified_by
            self.results[image_id].verified_at = datetime.now().isoformat()
            self._save_results()


# Convenience function
def cross_reference_anomaly(
    image_id: int,
    image_path: str,
    ra: float = None,
    dec: float = None,
) -> CrossReferenceResult:
    """Quick cross-reference of a single anomaly."""
    xref = CatalogCrossReference()
    return xref.cross_reference(image_id, image_path, ra, dec)
