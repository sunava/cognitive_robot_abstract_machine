from dataclasses import dataclass

from rclpy.duration import Duration
from rclpy.qos import QoSProfile

try:
    from rclpy_message_converter.message_converter import (
        convert_ros_message_to_dictionary,
        convert_dictionary_to_ros_message,
    )
except ModuleNotFoundError:
    convert_ros_message_to_dictionary = None
    convert_dictionary_to_ros_message = None
from typing_extensions import Dict, Type, Any

from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.adapters.json_serializer import (
    ExternalClassJSONSerializer,
    to_json,
    from_json,
)
from krrood.utils import get_full_class_name
from semantic_digital_twin.adapters.ros.utils import is_ros2_message_class


def _is_ros2_message_instance(obj: Any) -> bool:
    return is_ros2_message_class(obj.__class__)


def _fallback_ros_message_to_dict(msg: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for field in getattr(msg, "__slots__", []):
        data[field] = _fallback_to_basic(getattr(msg, field))
    return data


def _fallback_to_basic(value: Any) -> Any:
    if _is_ros2_message_instance(value):
        return _fallback_ros_message_to_dict(value)
    if isinstance(value, (list, tuple)):
        return [_fallback_to_basic(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    return value


def _fallback_from_basic(value: Any, current: Any) -> Any:
    if _is_ros2_message_instance(current) and isinstance(value, dict):
        return _fallback_dict_to_ros_message(current.__class__, value)
    if isinstance(current, (bytes, bytearray)) and isinstance(value, list):
        return type(current)(value)
    if isinstance(current, (list, tuple)) and isinstance(value, list):
        if current:
            prototype = current[0]
            if _is_ros2_message_instance(prototype):
                return [
                    (
                        _fallback_dict_to_ros_message(prototype.__class__, v)
                        if isinstance(v, dict)
                        else v
                    )
                    for v in value
                ]
        return [_fallback_to_basic(v) if isinstance(v, dict) else v for v in value]
    return value


def _fallback_dict_to_ros_message(clazz: Type, data: Dict[str, Any]) -> Any:
    msg = clazz()
    for field in getattr(msg, "__slots__", []):
        key = field
        if key not in data and key.startswith("_") and key[1:] in data:
            key = key[1:]
        if key not in data:
            continue
        current = getattr(msg, field)
        setattr(msg, field, _fallback_from_basic(data[key], current))
    return msg


if convert_ros_message_to_dictionary is None:

    def convert_ros_message_to_dictionary(msg: Any) -> Dict[str, Any]:
        return _fallback_ros_message_to_dict(msg)


if convert_dictionary_to_ros_message is None:

    def convert_dictionary_to_ros_message(
        clazz: Type, data: Dict[str, Any], **kwargs
    ) -> Any:
        return _fallback_dict_to_ros_message(clazz, data)


@dataclass
class Ros2MessageJSONSerializer(ExternalClassJSONSerializer[None]):
    """
    Json serializer for ROS2 messages.
    Since there is no common superclass for ROS2 messages, we need to rely on checking class fields instead.
    That's also why T is set to None.
    """

    @classmethod
    def to_json(cls, obj: Any) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(obj.__class__),
            "data": convert_ros_message_to_dictionary(obj),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Any:
        return convert_dictionary_to_ros_message(clazz, data["data"], **kwargs)

    @classmethod
    def matches_generic_type(cls, clazz: Type):
        return is_ros2_message_class(clazz)


@dataclass
class QoSProfileJSONSerializer(ExternalClassJSONSerializer[QoSProfile]):
    """
    A serializer class for converting a QoSProfile instance to and from JSON format.
    All fields of the QoSProfile are saved in its `__slots__` attribute with a `_` prefix.
    """

    @classmethod
    def to_json(cls, obj: QoSProfile) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(obj.__class__),
            **{
                field_name: to_json(getattr(obj, field_name))
                for field_name in obj.__slots__
            },
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type[QoSProfile], **kwargs) -> Any:
        return clazz(
            **{
                field_name[1:]: from_json(data[field_name])
                for field_name in clazz.__slots__
            }
        )


@dataclass
class DurationJSONSerializer(ExternalClassJSONSerializer[Duration]):
    """
    Serializer for converting Duration objects to and from JSON format.
    """

    @classmethod
    def to_json(cls, obj: Duration) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(obj.__class__),
            "nanoseconds": obj.nanoseconds,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], clazz: Type, **kwargs) -> Any:
        return clazz(nanoseconds=data["nanoseconds"])
