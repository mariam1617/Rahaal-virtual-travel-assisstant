import requests
from src.config import settings
from src.services.location import LocationResponse

def reverse_geocode(lat: float, lon: float) -> LocationResponse:
    """
    Calls the OpenMap reverse-geocoding API to convert coordinates into a structured location.
    """
    # TODO (Problem 5): A geocoding failure currently leaks into user answers.
    # 1. On error, reverse_geocode returns formatted_address="Location unavailable", which leaks into downstream
    #    nodes causing them to print "Emergency Assistance for Location unavailable".
    # 2. On success, the raw long display_name becomes the location. Need structured fields:
    #    governorate (address.state), neighborhood, country code, lat/lon.
    # 3. Add 'accept-language': 'en' to headers/params for consistent English localization.
    # 4. Add a valid contact email in the User-Agent header (required by OpenStreetMap/Nominatim usage policy).
    # 5. Add a local cache of successful lookups indexed by rounded coordinates (e.g. round to 3 decimal places)
    #    because the public Nominatim server expects low request rates (max 1 req/sec).
    url = "https://nominatim.openstreetmap.org/reverse"
    
    # OpenStreetMap/Nominatim uses format=json. 
    # Adjust parameters based on the specific OpenMap service you are using.
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        # Some mapping APIs require the key in the query params or headers
        # "api_key": settings.openmap_api_key 
    }
    
    headers = {
        "User-Agent": settings.project_name
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        
        address = data.get("address", {})
        
        return LocationResponse(
            city=address.get("city") or address.get("town") or address.get("village"),
            country=address.get("country"),
            formatted_address=data.get("display_name", "Unknown location")
        )
        
    except requests.exceptions.RequestException as e:
        # Return a safe fallback object so the graph can handle the error gracefully
        return LocationResponse(
            formatted_address="Location unavailable",
            error=str(e)
        )