"""
Google Earth Engine Bridge — Geospatial reality layer for E.D.I.T.H.
======================================================================
Uses the Earth Engine Python API (ee) for satellite imagery, land-use
analysis, and geospatial verification of research claims.

Capabilities:
  - Pull multi-year satellite imagery for any coordinates
  - Analyze land-use change over time (NDVI, built-up area, crop cover)
  - Verify physical claims from papers (buildings, infrastructure)
  - Generate geospatial counterfactuals for causal analysis
  - Export heatmaps and statistics for welfare/policy research

Requires: GOOGLE_EARTH_ENGINE_KEY (service account JSON path or key)
          or authenticated via `earthengine authenticate`
"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.google_earth_bridge")


class GoogleEarthBridge:
    """Google Earth Engine connector for geospatial policy research."""

    def __init__(self, service_account_key: str = ""):
        self._key_path = service_account_key or os.environ.get("GOOGLE_EARTH_ENGINE_KEY", "")
        self._project = os.environ.get("GOOGLE_EARTH_ENGINE_PROJECT", "")
        self._ee = None
        self._initialized = False

    def _ensure_ee(self) -> bool:
        """Lazy-initialize the Earth Engine API."""
        if self._initialized:
            return self._ee is not None
        self._initialized = True
        try:
            import ee
            self._ee = ee

            if self._key_path and os.path.isfile(self._key_path):
                # Service account authentication
                credentials = ee.ServiceAccountCredentials(None, self._key_path)
                ee.Initialize(credentials, project=self._project or None)
            elif self._key_path:
                # Inline JSON key
                import json as _json
                creds_dict = _json.loads(self._key_path) if self._key_path.startswith("{") else None
                if creds_dict:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                        _json.dump(creds_dict, f)
                        f.flush()
                        credentials = ee.ServiceAccountCredentials(None, f.name)
                        ee.Initialize(credentials, project=self._project or None)
                        os.unlink(f.name)
                else:
                    ee.Initialize(project=self._project or None)
            else:
                # Default credentials (user ran `earthengine authenticate`)
                ee.Initialize(project=self._project or None)

            log.info("Google Earth Engine initialized successfully")
            return True
        except ImportError:
            log.warning("earthengine-api not installed — run: pip install earthengine-api")
            self._ee = None
            return False
        except Exception as e:
            log.warning(f"Earth Engine initialization failed: {e}")
            self._ee = None
            return False

    # ── Core: Land-Use Analysis ──────────────────────────────────

    def analyze_land_use(self, lat: float, lon: float, radius_m: int = 5000,
                         start_year: int = 2015, end_year: int = 2025) -> dict:
        """Analyze land-use change over time at a location.

        Uses Landsat/Sentinel imagery to compute NDVI, built-up area,
        and vegetation trends.

        Args:
            lat, lon: Center coordinates
            radius_m: Analysis radius in meters (default 5km)
            start_year, end_year: Time range
        """
        if not self._ensure_ee():
            return {"error": "Earth Engine not available"}
        ee = self._ee

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(radius_m)

        results = {"location": {"lat": lat, "lon": lon, "radius_m": radius_m}, "years": {}}

        for year in range(start_year, end_year + 1):
            try:
                # Use Landsat 8/9 surface reflectance
                collection = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                              .filterBounds(region)
                              .filterDate(f"{year}-01-01", f"{year}-12-31")
                              .median())

                # NDVI = (NIR - Red) / (NIR + Red) — vegetation health
                ndvi = collection.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
                ndvi_stats = ndvi.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                    geometry=region, scale=30, maxPixels=1e8,
                ).getInfo()

                # Built-up index (NDBI) = (SWIR - NIR) / (SWIR + NIR)
                ndbi = collection.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
                ndbi_stats = ndbi.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=30, maxPixels=1e8,
                ).getInfo()

                results["years"][str(year)] = {
                    "ndvi_mean": round(ndvi_stats.get("NDVI_mean", 0) or 0, 4),
                    "ndvi_std": round(ndvi_stats.get("NDVI_stdDev", 0) or 0, 4),
                    "ndbi_mean": round(ndbi_stats.get("NDBI", 0) or 0, 4),
                    "vegetation_trend": "increasing" if (ndvi_stats.get("NDVI_mean", 0) or 0) > 0.3 else "declining",
                    "urbanization": "high" if (ndbi_stats.get("NDBI", 0) or 0) > 0.1 else "low",
                }
            except Exception as e:
                results["years"][str(year)] = {"error": str(e)}

        # Compute trend
        ndvi_values = [y.get("ndvi_mean", 0) for y in results["years"].values() if isinstance(y.get("ndvi_mean"), (int, float))]
        if len(ndvi_values) >= 2:
            trend = ndvi_values[-1] - ndvi_values[0]
            results["ndvi_trend"] = round(trend, 4)
            results["interpretation"] = (
                "Vegetation increasing — area greening (possible agriculture or parks)"
                if trend > 0.05 else
                "Vegetation declining — possible urbanization or land degradation"
                if trend < -0.05 else
                "Stable vegetation — no major land-use change detected"
            )

        return results

    # ── Geospatial Audit ─────────────────────────────────────────

    def audit_location(self, lat: float, lon: float, claim: str = "",
                       year_claimed: int = 2024) -> dict:
        """Verify a physical claim about a location using satellite data.

        E.g., verify that a "community center" still exists at given coordinates,
        or that a "food desert" claim is supported by land-use data.
        """
        if not self._ensure_ee():
            return {"error": "Earth Engine not available"}
        ee = self._ee

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(1000)  # 1km radius for detailed audit

        audit = {
            "location": {"lat": lat, "lon": lon},
            "claim": claim,
            "year_claimed": year_claimed,
        }

        try:
            # Get the most recent Sentinel-2 image
            recent = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                      .filterBounds(region)
                      .filterDate(f"{year_claimed}-01-01", f"{year_claimed}-12-31")
                      .sort("CLOUDY_PIXEL_PERCENTAGE")
                      .first())

            # NDVI for vegetation
            ndvi = recent.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndvi_val = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e8,
            ).getInfo()

            # Built-up area indicator
            ndbi = recent.normalizedDifference(["B11", "B8"]).rename("NDBI")
            ndbi_val = ndbi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e8,
            ).getInfo()

            ndvi_score = ndvi_val.get("NDVI", 0) or 0
            ndbi_score = ndbi_val.get("NDBI", 0) or 0

            audit["satellite_data"] = {
                "ndvi": round(ndvi_score, 4),
                "ndbi": round(ndbi_score, 4),
                "land_type": (
                    "dense_vegetation" if ndvi_score > 0.5 else
                    "moderate_vegetation" if ndvi_score > 0.2 else
                    "built_up" if ndbi_score > 0.1 else
                    "barren_or_water"
                ),
                "has_structures": ndbi_score > 0.05,
                "is_green_space": ndvi_score > 0.3,
            }

            # Compare with older data to detect demolition/change
            try:
                older = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                         .filterBounds(region)
                         .filterDate(f"{year_claimed - 3}-01-01", f"{year_claimed - 3}-12-31")
                         .sort("CLOUDY_PIXEL_PERCENTAGE")
                         .first())
                old_ndbi = older.normalizedDifference(["B11", "B8"]).rename("NDBI")
                old_ndbi_val = old_ndbi.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e8,
                ).getInfo()
                old_ndbi_score = old_ndbi_val.get("NDBI", 0) or 0

                change = ndbi_score - old_ndbi_score
                audit["change_detected"] = {
                    "ndbi_change": round(change, 4),
                    "interpretation": (
                        "New construction detected" if change > 0.1 else
                        "Possible demolition/clearing" if change < -0.1 else
                        "No significant structural change"
                    ),
                }
            except Exception:
                audit["change_detected"] = {"note": "Historical comparison unavailable"}

            # Food desert proxy: check for nearby green space / agriculture
            if "food" in claim.lower() or "desert" in claim.lower() or "grocery" in claim.lower():
                wider_region = point.buffer(5000)
                wide_ndvi = ndvi.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=wider_region, scale=30, maxPixels=1e8,
                ).getInfo()
                audit["food_desert_analysis"] = {
                    "area_vegetation": round(wide_ndvi.get("NDVI", 0) or 0, 4),
                    "likely_food_desert": (wide_ndvi.get("NDVI", 0) or 0) < 0.15 and ndbi_score > 0.1,
                    "note": "Low vegetation + high built-up index suggests limited agricultural access",
                }

        except Exception as e:
            audit["error"] = str(e)

        return audit

    # ── Geospatial Counterfactual ────────────────────────────────

    def counterfactual_analysis(self, lat: float, lon: float, radius_m: int = 10000,
                                 policy_variable: str = "", baseline_year: int = 2020,
                                 projection_years: int = 5) -> dict:
        """Run a geospatial counterfactual — compare treated vs. matched control.

        Instead of a naive coordinate offset, finds a control area with similar
        baseline characteristics (NDVI/NDBI) by sampling in 4 cardinal directions
        and selecting the best match.
        """
        if not self._ensure_ee():
            return {"error": "Earth Engine not available"}
        ee = self._ee

        # Get treated area baseline
        treated = self.analyze_land_use(lat, lon, radius_m=radius_m,
                                        start_year=baseline_year, end_year=baseline_year + projection_years)
        treated_baseline = treated.get("years", {}).get(str(baseline_year), {})
        treated_ndvi_base = treated_baseline.get("ndvi_mean", 0)
        treated_ndbi_base = treated_baseline.get("ndbi_mean", 0)

        # Sample 4 candidate control areas at ~3x radius in cardinal directions
        offset_deg = (radius_m * 3) / 111000  # rough meters-to-degrees
        candidates = [
            ("north", lat + offset_deg, lon),
            ("south", lat - offset_deg, lon),
            ("east", lat, lon + offset_deg),
            ("west", lat, lon - offset_deg),
        ]

        best_control = None
        best_similarity = float("inf")
        best_direction = ""

        for direction, c_lat, c_lon in candidates:
            try:
                c_data = self.analyze_land_use(c_lat, c_lon, radius_m=radius_m,
                                               start_year=baseline_year, end_year=baseline_year + projection_years)
                c_baseline = c_data.get("years", {}).get(str(baseline_year), {})
                c_ndvi = c_baseline.get("ndvi_mean", 0)
                c_ndbi = c_baseline.get("ndbi_mean", 0)

                # Similarity = Euclidean distance in (NDVI, NDBI) space
                similarity = ((c_ndvi - treated_ndvi_base) ** 2 + (c_ndbi - treated_ndbi_base) ** 2) ** 0.5

                if similarity < best_similarity:
                    best_similarity = similarity
                    best_control = c_data
                    best_direction = direction
            except Exception:
                continue

        if not best_control:
            return {"error": "Could not find suitable control area"}

        treated_trend = treated.get("ndvi_trend", 0)
        control_trend = best_control.get("ndvi_trend", 0)
        diff = treated_trend - control_trend

        c_lat_used = candidates[["north", "south", "east", "west"].index(best_direction)][1]
        c_lon_used = candidates[["north", "south", "east", "west"].index(best_direction)][2]

        return {
            "treated_area": {"lat": lat, "lon": lon, "trend": treated_trend,
                             "baseline_ndvi": treated_ndvi_base, "baseline_ndbi": treated_ndbi_base},
            "control_area": {"lat": c_lat_used, "lon": c_lon_used, "trend": control_trend,
                             "direction": best_direction, "similarity_score": round(best_similarity, 4),
                             "match_quality": "excellent" if best_similarity < 0.05 else "good" if best_similarity < 0.15 else "fair"},
            "difference_in_differences": round(diff, 4),
            "policy_variable": policy_variable,
            "interpretation": (
                f"Matched control found {best_direction} of treated area "
                f"(baseline similarity: {best_similarity:.4f}). "
                f"The treated area shows {'greater' if diff > 0 else 'less'} vegetation change "
                f"than the matched control (DiD={diff:.4f}). "
                + ("This suggests the policy had a measurable positive environmental impact. " if diff > 0.02 else "")
                + ("This suggests the policy may have had negative environmental effects. " if diff < -0.02 else "")
                + ("No significant difference detected between areas. " if abs(diff) <= 0.02 else "")
            ),
        }

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> dict:
        """Check if Earth Engine is configured and reachable."""
        try:
            import ee  # noqa: F401
            has_package = True
        except ImportError:
            has_package = False

        if not has_package:
            return {
                "available": False, "configured": False,
                "reason": "earthengine-api not installed (pip install earthengine-api)",
            }

        has_key = bool(self._key_path) or bool(self._project)

        if self._ensure_ee():
            return {
                "available": True, "configured": True,
                "project": self._project or "(default)",
                "note": "Geospatial audit + land-use analysis ready",
            }
        return {
            "available": False, "configured": has_key,
            "reason": "Earth Engine authentication failed — run `earthengine authenticate` or set GOOGLE_EARTH_ENGINE_KEY",
        }
