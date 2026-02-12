from django.apps import AppConfig
from django.db.models.signals import post_migrate


def register_cleanup_schedule(sender, **kwargs):
    """Register the periodic stuck-task cleanup schedule with Django-Q2.

    Uses the post_migrate signal so it runs only after all tables exist,
    avoiding the 'database access during app initialization' warning.
    """
    from django.db import OperationalError, ProgrammingError

    try:
        from django_q.models import Schedule

        Schedule.objects.update_or_create(
            name='cleanup_stuck_tasks',
            defaults={
                'func': 'council_app.tasks.cleanup_stuck_tasks',
                'schedule_type': Schedule.MINUTES,
                'minutes': 10,  # Run every 10 minutes
                'repeats': -1,  # Repeat forever
            },
        )
    except (OperationalError, ProgrammingError, Exception):
        pass


class CouncilAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'council_app'

    def ready(self):
        post_migrate.connect(register_cleanup_schedule, sender=self)
