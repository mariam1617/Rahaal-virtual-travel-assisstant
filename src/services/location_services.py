import requests
from src.config import settings
from src.services.location import LocationResponse

def reverse_geocode(lat: float, lon: float) -> LocationResponse:
    """
    Calls the OpenMap reverse-geocoding API to convert coordinates into a structured location.
    """
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