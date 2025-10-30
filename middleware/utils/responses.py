from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

# Define a type variable for generic response models
T = TypeVar('T')

class StatusCode(int, Enum):
    """HTTP Status codes used in the API"""
    # Success codes
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    # Client error codes
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    
    # Server error codes
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    SERVICE_UNAVAILABLE = 503


class ErrorDetail(BaseModel):
    """Model for detailed error information"""
    loc: Optional[List[str]] = None
    msg: str
    type: str


class ResponseModel(BaseModel, Generic[T]):
    """Standard response model for all API endpoints"""
    success: bool
    message: str
    data: Optional[T] = None
    errors: Optional[List[ErrorDetail]] = None
    
    @classmethod
    def success_response(cls, data: Any = None, message: str = "Operation successful"):
        """Create a success response"""
        return cls(success=True, message=message, data=data)
    
    @classmethod
    def error_response(cls, message: str, errors: List[ErrorDetail] = None):
        """Create an error response"""
        return cls(success=False, message=message, errors=errors)


def create_response(
    data: Any = None, 
    message: str = "Operation successful", 
    status_code: StatusCode = StatusCode.OK
) -> JSONResponse:
    """
    Create a standardized JSON response
    
    Args:
        data: The data to return
        message: A message describing the result
        status_code: HTTP status code
        
    Returns:
        JSONResponse with standardized format
    """
    response_content = ResponseModel.success_response(data=data, message=message).dict()
    return JSONResponse(content=response_content, status_code=status_code)


def create_error_response(
    message: str,
    status_code: StatusCode = StatusCode.BAD_REQUEST,
    errors: List[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Create a standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        errors: List of detailed error information
        
    Returns:
        JSONResponse with standardized error format
    """
    error_details = None
    if errors:
        error_details = [ErrorDetail(**error) for error in errors]
    
    response_content = ResponseModel.error_response(message=message, errors=error_details).dict()
    return JSONResponse(content=response_content, status_code=status_code)


# Common exceptions that can be raised in route handlers
class APIException(HTTPException):
    """Base exception for API errors"""
    def __init__(
        self, 
        status_code: int, 
        detail: str, 
        errors: List[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.errors = errors


class BadRequestException(APIException):
    """Exception for 400 Bad Request errors"""
    def __init__(self, detail: str = "Bad request", errors: List[Dict[str, Any]] = None):
        super().__init__(status_code=StatusCode.BAD_REQUEST, detail=detail, errors=errors)


class NotFoundException(APIException):
    """Exception for 404 Not Found errors"""
    def __init__(self, detail: str = "Resource not found", errors: List[Dict[str, Any]] = None):
        super().__init__(status_code=StatusCode.NOT_FOUND, detail=detail, errors=errors)


class InternalServerErrorException(APIException):
    """Exception for 500 Internal Server Error"""
    def __init__(self, detail: str = "Internal server error", errors: List[Dict[str, Any]] = None):
        super().__init__(status_code=StatusCode.INTERNAL_SERVER_ERROR, detail=detail, errors=errors)