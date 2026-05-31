import os

os.environ.setdefault('ALLOW_PUBLIC_REGISTER', 'false')
os.environ.setdefault('MYSQL_PASSWORD', 'dummy')

from app import app  # noqa: E402


def test_register_page_redirects_when_public_register_disabled():
    with app.test_client() as c:
        r = c.get('/admin/register', follow_redirects=False)
        assert r.status_code in (301, 302)
        assert '/admin/login' in (r.headers.get('Location') or '')


def test_register_api_disabled_when_public_register_closed():
    with app.test_client() as c:
        r = c.post('/api/admin/register', json={'username': 'u1', 'password': '123456'})
        assert r.status_code == 403
        data = r.get_json() or {}
        assert data.get('error') == 'forbidden'


def test_admin_records_requires_login():
    with app.test_client() as c:
        r = c.get('/api/admin/records')
        assert r.status_code == 401
        data = r.get_json() or {}
        assert data.get('error') == 'unauthorized'


def test_admin_summary_requires_login():
    with app.test_client() as c:
        r = c.get('/api/admin/summary')
        assert r.status_code == 401
        data = r.get_json() or {}
        assert data.get('error') == 'unauthorized'
