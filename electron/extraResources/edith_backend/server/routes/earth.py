"""
§EARTH: Google Earth Engine Integration
Provides geospatial research capabilities:
  /api/earth/kml-export   — Export paper locations as KML for Google Earth
  /api/earth/satellite     — Get satellite/terrain context for a location
  /api/earth/regions       — Country/region polygons with paper density
  /api/earth/overlay       — Generate research density overlay (GeoJSON)
  /api/earth/geocode       — Geocode a paper's country to coordinates
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
import logging
import os
import json

log = logging.getLogger("edith.routes.earth")
router = APIRouter()


# ── Country coordinate database ──────────────────────────────────
COUNTRY_COORDS = {
    "United States": {"lat": 39.83, "lng": -98.58, "zoom": 4},
    "Mexico": {"lat": 23.63, "lng": -102.55, "zoom": 5},
    "Brazil": {"lat": -14.24, "lng": -51.93, "zoom": 4},
    "United Kingdom": {"lat": 55.38, "lng": -3.44, "zoom": 5},
    "France": {"lat": 46.23, "lng": 2.21, "zoom": 5},
    "Germany": {"lat": 51.17, "lng": 10.45, "zoom": 5},
    "Spain": {"lat": 40.46, "lng": -3.75, "zoom": 5},
    "Italy": {"lat": 41.87, "lng": 12.57, "zoom": 5},
    "India": {"lat": 20.59, "lng": 78.96, "zoom": 4},
    "China": {"lat": 35.86, "lng": 104.20, "zoom": 4},
    "Japan": {"lat": 36.20, "lng": 138.25, "zoom": 5},
    "South Korea": {"lat": 35.91, "lng": 127.77, "zoom": 6},
    "Australia": {"lat": -25.27, "lng": 133.78, "zoom": 4},
    "Canada": {"lat": 56.13, "lng": -106.35, "zoom": 3},
    "Russia": {"lat": 61.52, "lng": 105.32, "zoom": 3},
    "South Africa": {"lat": -30.56, "lng": 22.94, "zoom": 5},
    "Nigeria": {"lat": 9.08, "lng": 8.68, "zoom": 5},
    "Kenya": {"lat": -0.02, "lng": 37.91, "zoom": 6},
    "Egypt": {"lat": 26.82, "lng": 30.80, "zoom": 5},
    "Argentina": {"lat": -38.42, "lng": -63.62, "zoom": 4},
    "Colombia": {"lat": 4.57, "lng": -74.30, "zoom": 5},
    "Chile": {"lat": -35.68, "lng": -71.54, "zoom": 4},
    "Peru": {"lat": -9.19, "lng": -75.02, "zoom": 5},
    "Guatemala": {"lat": 15.78, "lng": -90.23, "zoom": 6},
    "Honduras": {"lat": 15.20, "lng": -86.24, "zoom": 6},
    "El Salvador": {"lat": 13.79, "lng": -88.90, "zoom": 7},
    "Nicaragua": {"lat": 12.87, "lng": -85.21, "zoom": 6},
    "Costa Rica": {"lat": 9.75, "lng": -83.75, "zoom": 7},
    "Panama": {"lat": 8.54, "lng": -80.78, "zoom": 7},
    "Cuba": {"lat": 21.52, "lng": -77.78, "zoom": 6},
    "Venezuela": {"lat": 6.42, "lng": -66.59, "zoom": 5},
    "Texas": {"lat": 31.97, "lng": -99.90, "zoom": 6},
    "Rural Texas": {"lat": 31.50, "lng": -100.44, "zoom": 7},
    "Washington DC": {"lat": 38.91, "lng": -77.04, "zoom": 10},
    "Global": {"lat": 20.0, "lng": 0.0, "zoom": 2},
}

# Country → region mapping for research context
RESEARCH_REGIONS = {
    "North America": ["United States", "Canada", "Mexico", "Texas", "Rural Texas", "Washington DC"],
    "Central America": ["Guatemala", "Honduras", "El Salvador", "Nicaragua", "Costa Rica", "Panama"],
    "South America": ["Brazil", "Argentina", "Colombia", "Chile", "Peru", "Venezuela"],
    "Caribbean": ["Cuba"],
    "Western Europe": ["United Kingdom", "France", "Germany", "Spain", "Italy"],
    "Eastern Europe": ["Russia"],
    "East Asia": ["China", "Japan", "South Korea"],
    "South Asia": ["India"],
    "Africa": ["South Africa", "Nigeria", "Kenya", "Egypt"],
    "Oceania": ["Australia"],
}


@router.post("/api/earth/kml-export", tags=["Earth"])
async def export_kml(request: Request):
    """Export paper locations as KML file for Google Earth."""
    body = await request.json()
    papers = body.get("papers", [])
    title = body.get("title", "E.D.I.T.H. Research Map")

    kml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'<name>{title}</name>',
        '<description>Research papers exported from E.D.I.T.H.</description>',
        '',
        '<!-- Styles -->',
        '<Style id="paper-pin"><IconStyle><color>ff4444ff</color>'
        '<scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>'
        '</IconStyle></Style>',
        '<Style id="cluster-pin"><IconStyle><color>ff00aaff</color>'
        '<scale>1.3</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/blu-circle.png</href></Icon>'
        '</IconStyle></Style>',
        '',
    ]

    placed = 0
    for paper in papers:
        country = paper.get("country", "")
        coords = COUNTRY_COORDS.get(country, {})
        if not coords:
            continue
        lat = coords["lat"] + (hash(paper.get("title", "")) % 100) / 500.0
        lng = coords["lng"] + (hash(paper.get("sha256", "")) % 100) / 500.0

        kml_parts.append('<Placemark>')
        kml_parts.append(f'  <name>{_xml_escape(paper.get("title", "Untitled"))}</name>')
        desc_lines = []
        if paper.get("author"):
            desc_lines.append(f'Author: {paper["author"]}')
        if paper.get("year"):
            desc_lines.append(f'Year: {paper["year"]}')
        if paper.get("topic"):
            desc_lines.append(f'Topic: {paper["topic"]}')
        if paper.get("method"):
            desc_lines.append(f'Method: {paper["method"]}')
        kml_parts.append(f'  <description>{_xml_escape(chr(10).join(desc_lines))}</description>')
        kml_parts.append('  <styleUrl>#paper-pin</styleUrl>')
        kml_parts.append(f'  <Point><coordinates>{lng},{lat},0</coordinates></Point>')
        kml_parts.append('</Placemark>')
        placed += 1

    kml_parts.extend(['</Document>', '</kml>'])
    kml_content = '\n'.join(kml_parts)

    return Response(
        content=kml_content,
        media_type="application/vnd.google-earth.kml+xml",
        headers={
            "Content-Disposition": f'attachment; filename="edith_papers.kml"',
            "X-Papers-Placed": str(placed),
        },
    )


@router.get("/api/earth/satellite", tags=["Earth"])
async def satellite_context(request: Request):
    """Get satellite/terrain context URL for a location (Google Earth Web link)."""
    country = request.query_params.get("country", "")
    lat = request.query_params.get("lat")
    lng = request.query_params.get("lng")

    if not lat or not lng:
        coords = COUNTRY_COORDS.get(country, COUNTRY_COORDS.get("Global", {}))
        lat = coords.get("lat", 20.0)
        lng = coords.get("lng", 0.0)
        zoom = coords.get("zoom", 4)
    else:
        lat, lng = float(lat), float(lng)
        zoom = 8

    google_earth_url = (
        f"https://earth.google.com/web/@{lat},{lng},0a,"
        f"{10000000 / (2 ** zoom)}d,35y,0h,0t,0r"
    )
    google_maps_url = (
        f"https://www.google.com/maps/@{lat},{lng},{zoom}z/data=!3m1!1e3"
    )

    return {
        "country": country,
        "lat": lat,
        "lng": lng,
        "zoom": zoom,
        "google_earth_url": google_earth_url,
        "google_maps_satellite_url": google_maps_url,
        "embed_url": f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom={zoom}&size=600x400&maptype=satellite&key=YOUR_API_KEY",
    }


@router.get("/api/earth/regions", tags=["Earth"])
async def research_regions():
    """Get research regions with paper density data."""
    # Build region data from library cache
    regions = []
    for region_name, countries in RESEARCH_REGIONS.items():
        country_data = []
        for c in countries:
            coords = COUNTRY_COORDS.get(c, {})
            if coords:
                country_data.append({
                    "name": c,
                    "lat": coords["lat"],
                    "lng": coords["lng"],
                    "zoom": coords["zoom"],
                })
        if country_data:
            avg_lat = sum(c["lat"] for c in country_data) / len(country_data)
            avg_lng = sum(c["lng"] for c in country_data) / len(country_data)
            regions.append({
                "region": region_name,
                "center": {"lat": avg_lat, "lng": avg_lng},
                "countries": country_data,
                "country_count": len(country_data),
            })
    return {"regions": regions, "total_countries": len(COUNTRY_COORDS)}


@router.post("/api/earth/overlay", tags=["Earth"])
async def research_overlay(request: Request):
    """Generate GeoJSON research density overlay for map rendering."""
    body = await request.json()
    papers = body.get("papers", [])
    overlay_type = body.get("type", "density")  # density, cluster, connections

    features = []
    country_counts: dict[str, int] = {}
    for paper in papers:
        country = paper.get("country", "Unknown")
        country_counts[country] = country_counts.get(country, 0) + 1

    for country, count in country_counts.items():
        coords = COUNTRY_COORDS.get(country)
        if not coords:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "name": country,
                "paper_count": count,
                "density": min(count / 10.0, 1.0),
                "color": _density_color(count),
            },
            "geometry": {
                "type": "Point",
                "coordinates": [coords["lng"], coords["lat"]],
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_papers": len(papers),
            "total_countries": len(country_counts),
            "overlay_type": overlay_type,
        },
    }


@router.get("/api/earth/geocode", tags=["Earth"])
async def geocode_country(request: Request):
    """Geocode a country name to coordinates for map placement."""
    country = request.query_params.get("country", "")
    coords = COUNTRY_COORDS.get(country)
    if coords:
        return {"ok": True, "country": country, **coords}
    # Fuzzy match
    lower = country.lower()
    for name, c in COUNTRY_COORDS.items():
        if lower in name.lower() or name.lower() in lower:
            return {"ok": True, "country": name, **c, "fuzzy_match": True}
    return {"ok": False, "error": f"Country '{country}' not found", "available": list(COUNTRY_COORDS.keys())}


def _xml_escape(s: str) -> str:
    """Escape XML special characters."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _density_color(count: int) -> str:
    """Return color based on paper count density."""
    if count >= 20:
        return "#dc2626"  # red — heavy
    elif count >= 10:
        return "#f97316"  # orange — moderate
    elif count >= 5:
        return "#eab308"  # yellow — some
    elif count >= 2:
        return "#22c55e"  # green — light
    else:
        return "#3b82f6"  # blue — sparse
