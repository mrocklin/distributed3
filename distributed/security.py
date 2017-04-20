from __future__ import print_function, division, absolute_import

import ssl

from . import config


_roles = ['client', 'scheduler', 'worker']

_tls_per_role_fields = ['key', 'cert']

_tls_fields = ['ca_file']

_fields = set(['tls_%s' % field for field in _tls_fields] +
              ['tls_%s_%s' % (role, field)
               for role in _roles
               for field in _tls_per_role_fields]
              )


def _field_to_config_key(field):
    return field.replace('_', '-')


class Security(object):
    """
    An object to gather and pass around security configuration.
    Default values are gathered from the global ``config`` object and
    can be overriden by constructor args.

    Supported fields:
        - tls_ca_file
        - tls_client_key
        - tls_client_cert
        - tls_scheduler_key
        - tls_scheduler_cert
        - tls_worker_key
        - tls_worker_cert
    """

    __slots__ = tuple(_fields)

    def __init__(self, **kwargs):
        self._init_from_dict(config)
        for k, v in kwargs.items():
            setattr(self, k, v)
        for k in _fields:
            if not hasattr(self, k):
                setattr(self, k, None)

    def _init_from_dict(self, d):
        """
        Initialize Security from nested dict.
        """
        self._init_fields_from_dict(d, 'tls', _tls_fields, _tls_per_role_fields)

    def _init_fields_from_dict(self, d, category,
                               fields, per_role_fields):
        d = d.get(category, {})
        for field in fields:
            k = _field_to_config_key(field)
            if k in d:
                setattr(self, '%s_%s' % (category, field), d[k])
        for role in _roles:
            dd = d.get(role, {})
            for field in per_role_fields:
                k = _field_to_config_key(field)
                if k in dd:
                    setattr(self, '%s_%s_%s' % (category, role, field), dd[k])

    def __repr__(self):
        items = sorted((k, getattr(self, k)) for k in _fields)
        return ("Security(" +
                ", ".join("%s=%r" % (k, v) for k, v in items if v is not None) +
                ")")

    def get_tls_config_for_role(self, role):
        """
        Return the TLS configuration for the given role, as a flat dict.
        """
        return self._get_config_for_role('tls', role, _tls_fields, _tls_per_role_fields)

    def _get_config_for_role(self, category, role, fields, per_role_fields):
        if role not in _roles:
            raise ValueError("unknown role %r" % (role,))
        d = {}
        for field in fields:
            k = '%s_%s' % (category, field)
            d[field] = getattr(self, k)
        for field in per_role_fields:
            k = '%s_%s_%s' % (category, role, field)
            d[field] = getattr(self, k)
        return d

    def _get_tls_context(self, tls, purpose):
        if tls.get('ca_file') and tls.get('cert'):
            ctx = ssl.create_default_context(purpose=purpose,
                                             cafile=tls['ca_file'])
            ctx.verify_mode = ssl.CERT_REQUIRED
            # We expect a dedicated CA for the cluster and people using
            # IP addresses rather than hostnames
            ctx.check_hostname = False
            ctx.load_cert_chain(tls['cert'], tls.get('key'))
            return ctx

    def get_connection_args(self, role):
        """
        Get the *connection_args* argument for a connect() call with
        the given *role*.
        """
        d = {}
        tls = self.get_tls_config_for_role(role)
        d['ssl_context'] = self._get_tls_context(tls, ssl.Purpose.SERVER_AUTH)
        return d

    def get_listen_args(self, role):
        """
        Get the *connection_args* argument for a listen() call with
        the given *role*.
        """
        d = {}
        tls = self.get_tls_config_for_role(role)
        d['ssl_context'] = self._get_tls_context(tls, ssl.Purpose.CLIENT_AUTH)
        return d
