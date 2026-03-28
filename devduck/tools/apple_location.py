"""📍 Apple Location Services via CoreLocation framework."""

from typing import Dict, Any
from strands import tool


@tool
def apple_location(
    action: str = "current",
    address: str = None,
    latitude: float = None,
    longitude: float = None,
) -> Dict[str, Any]:
    """📍 Location services — current position, geocoding, reverse geocoding.

    Args:
        action: Action to perform:
            - "current": Get current location (lat/lon/accuracy)
            - "geocode": Address → coordinates
            - "reverse": Coordinates → address
            - "status": Authorization and service status
        address: Address string for geocoding
        latitude: Latitude for reverse geocoding
        longitude: Longitude for reverse geocoding

    Returns:
        Dict with location data
    """
    try:
        import CoreLocation
        from Foundation import NSRunLoop, NSDate, NSDefaultRunLoopMode
    except ImportError:
        return {"status": "error", "content": [{"text": "Install: pip install pyobjc-framework-CoreLocation"}]}

    if action == "current":
        return _get_current_location()
    elif action == "geocode":
        if not address:
            return {"status": "error", "content": [{"text": "address required for geocode"}]}
        return _geocode(address)
    elif action == "reverse":
        if latitude is None or longitude is None:
            return {"status": "error", "content": [{"text": "latitude and longitude required for reverse geocode"}]}
        return _reverse_geocode(latitude, longitude)
    elif action == "status":
        return _location_status()
    else:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: current, geocode, reverse, status"}]}


def _location_status():
    """Check location services status."""
    try:
        import CoreLocation

        enabled = CoreLocation.CLLocationManager.locationServicesEnabled()
        auth = CoreLocation.CLLocationManager.authorizationStatus()

        auth_map = {
            0: "Not Determined",
            1: "Restricted",
            2: "Denied",
            3: "Authorized Always",
            4: "Authorized When In Use",
        }

        lines = [
            "📍 Location Services:",
            f"  Enabled: {'✅' if enabled else '❌'} {enabled}",
            f"  Authorization: {auth_map.get(auth, f'Unknown ({auth})')}",
        ]

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Status error: {e}"}]}


def _get_current_location():
    """Get current location using CoreLocation."""
    try:
        import CoreLocation
        import objc
        from Foundation import NSObject, NSRunLoop, NSDate, NSDefaultRunLoopMode
        import time

        class LocationDelegate(NSObject):
            def init(self):
                self = objc.super(LocationDelegate, self).init()
                self.location = None
                self.error = None
                self.done = False
                return self

            def locationManager_didUpdateLocations_(self, manager, locations):
                if locations and len(locations) > 0:
                    self.location = locations[-1]
                self.done = True

            def locationManager_didFailWithError_(self, manager, error):
                self.error = str(error)
                self.done = True

            def locationManager_didChangeAuthorizationStatus_(self, manager, status):
                if status in (3, 4):  # Authorized
                    manager.requestLocation()

        delegate = LocationDelegate.alloc().init()
        manager = CoreLocation.CLLocationManager.alloc().init()
        manager.setDelegate_(delegate)
        manager.setDesiredAccuracy_(CoreLocation.kCLLocationAccuracyBest)

        # Request authorization and location
        manager.requestLocation()

        deadline = time.time() + 15
        while not delegate.done and time.time() < deadline:
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.2)
            )

        if delegate.error:
            return {"status": "error", "content": [{"text": f"Location error: {delegate.error}"}]}

        if not delegate.location:
            return {"status": "error", "content": [{"text": "Location request timed out. Check System Preferences → Privacy → Location Services."}]}

        loc = delegate.location
        lat = loc.coordinate().latitude
        lon = loc.coordinate().longitude
        alt = loc.altitude()
        acc = loc.horizontalAccuracy()

        lines = [
            "📍 Current Location:",
            f"  Latitude:  {lat:.6f}",
            f"  Longitude: {lon:.6f}",
            f"  Altitude:  {alt:.1f}m",
            f"  Accuracy:  ±{acc:.0f}m",
            f"  Maps: https://maps.apple.com/?ll={lat},{lon}",
        ]

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Location error: {e}"}]}


def _geocode(address):
    """Convert address to coordinates."""
    try:
        import CoreLocation
        from Foundation import NSRunLoop, NSDate, NSDefaultRunLoopMode
        import time

        geocoder = CoreLocation.CLGeocoder.alloc().init()
        results = []
        done = [False]
        error_msg = [None]

        def callback(placemarks, error):
            if placemarks:
                results.extend(placemarks)
            if error:
                error_msg[0] = str(error)
            done[0] = True

        geocoder.geocodeAddressString_completionHandler_(address, callback)

        deadline = time.time() + 15
        while not done[0] and time.time() < deadline:
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.2)
            )

        if error_msg[0]:
            return {"status": "error", "content": [{"text": f"Geocode error: {error_msg[0]}"}]}

        if not results:
            return {"status": "success", "content": [{"text": f"📍 No results for '{address}'."}]}

        lines = [f"📍 Geocode results for '{address}':\n"]
        for pm in results:
            loc = pm.location()
            lat = loc.coordinate().latitude
            lon = loc.coordinate().longitude
            name = pm.name() or ""
            locality = pm.locality() or ""
            country = pm.country() or ""

            lines.append(f"  📌 {name}")
            lines.append(f"     {locality}, {country}")
            lines.append(f"     {lat:.6f}, {lon:.6f}")
            lines.append(f"     🗺️  https://maps.apple.com/?ll={lat},{lon}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Geocode error: {e}"}]}


def _reverse_geocode(lat, lon):
    """Convert coordinates to address."""
    try:
        import CoreLocation
        from Foundation import NSRunLoop, NSDate, NSDefaultRunLoopMode
        import time

        geocoder = CoreLocation.CLGeocoder.alloc().init()
        location = CoreLocation.CLLocation.alloc().initWithLatitude_longitude_(lat, lon)

        results = []
        done = [False]
        error_msg = [None]

        def callback(placemarks, error):
            if placemarks:
                results.extend(placemarks)
            if error:
                error_msg[0] = str(error)
            done[0] = True

        geocoder.reverseGeocodeLocation_completionHandler_(location, callback)

        deadline = time.time() + 15
        while not done[0] and time.time() < deadline:
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.2)
            )

        if error_msg[0]:
            return {"status": "error", "content": [{"text": f"Reverse geocode error: {error_msg[0]}"}]}

        if not results:
            return {"status": "success", "content": [{"text": f"📍 No address for ({lat}, {lon})."}]}

        pm = results[0]
        parts = []
        if pm.subThoroughfare(): parts.append(str(pm.subThoroughfare()))
        if pm.thoroughfare(): parts.append(str(pm.thoroughfare()))
        if pm.locality(): parts.append(str(pm.locality()))
        if pm.administrativeArea(): parts.append(str(pm.administrativeArea()))
        if pm.postalCode(): parts.append(str(pm.postalCode()))
        if pm.country(): parts.append(str(pm.country()))

        lines = [
            f"📍 Reverse Geocode ({lat:.4f}, {lon:.4f}):",
            f"  Address: {', '.join(parts)}",
            f"  🗺️  https://maps.apple.com/?ll={lat},{lon}",
        ]

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Reverse geocode error: {e}"}]}
