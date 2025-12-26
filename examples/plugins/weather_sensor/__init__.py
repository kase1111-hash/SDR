"""
Weather Sensor Protocol Plugin.

Exports the WeatherSensorPlugin class for the plugin system.
"""

from .plugin import WeatherSensorPlugin, WeatherData, SensorType

__all__ = ["WeatherSensorPlugin", "WeatherData", "SensorType"]
