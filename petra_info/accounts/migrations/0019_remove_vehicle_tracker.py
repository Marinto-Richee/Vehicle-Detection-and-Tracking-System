# Generated by Django 4.2.16 on 2024-09-28 16:40

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0018_alter_detection_tracker"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="vehicle",
            name="tracker",
        ),
    ]
