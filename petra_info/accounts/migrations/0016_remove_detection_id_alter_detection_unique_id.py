# Generated by Django 4.2.16 on 2024-09-28 15:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0015_detection_unique_id"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="detection",
            name="id",
        ),
        migrations.AlterField(
            model_name="detection",
            name="unique_id",
            field=models.CharField(
                default="Unknown", max_length=100, primary_key=True, serialize=False
            ),
        ),
    ]