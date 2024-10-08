import logging
from functools import wraps
from typing import Any, Callable, Dict

from quart import abort, current_app, request

from config import CONFIG_AUTH_CLIENT, CONFIG_SEARCH_CLIENT, CONFIG_AUTH_ALLOWED_GROUPS
from core.authentication import AuthError
from error import error_response


def authenticated_path(route_fn: Callable[[str, Dict[str, Any]], Any]):
    """
    Decorator for routes that request a specific file that might require access control enforcement
    """

    @wraps(route_fn)
    async def auth_handler(path=""):
        # If authentication is enabled, validate the user can access the file
        auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
        allowed_groups = current_app.config[CONFIG_AUTH_ALLOWED_GROUPS]
        search_client = current_app.config[CONFIG_SEARCH_CLIENT]
        authorized = False
        try:
            auth_claims = await auth_helper.get_auth_claims_if_enabled(request.headers)
            authorized = await auth_helper.check_group_auth(allowed_groups, auth_claims)
            if authorized:
                authorized = await auth_helper.check_path_auth(path, auth_claims, search_client)
        except AuthError as error:
            abort(error.status_code)
        except Exception as error:
            logging.exception("Problem checking path auth %s", error)
            return error_response(error, route="/content")

        if not authorized:
            abort(403)

        return await route_fn(path, auth_claims)

    return auth_handler


def authenticated(route_fn: Callable[[Dict[str, Any]], Any]):
    """
    Decorator for routes that might require access control. Unpacks Authorization header information into an auth_claims dictionary
    """

    @wraps(route_fn)
    async def auth_handler():
        auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
        allowed_groups = current_app.config[CONFIG_AUTH_ALLOWED_GROUPS]
        authorized = False
        try:
            auth_claims = await auth_helper.get_auth_claims_if_enabled(request.headers)
            authorized = await auth_helper.check_group_auth(allowed_groups, auth_claims)
        except AuthError as error:
            abort(error.status_code)

        if not authorized:
            abort(403)

        return await route_fn(auth_claims)

    return auth_handler
