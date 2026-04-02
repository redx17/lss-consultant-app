import reflex as rx

config = rx.Config(
    app_name="LM_STUDIO_LEAN_SIX_SIGMA",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)