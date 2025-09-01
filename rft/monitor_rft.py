from openai import OpenAI
import os
import time
import datetime as dt
from typing import Optional


class OpenAIJobManager:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def list_jobs(self, limit: int = 20) -> None:
        """List all fine-tuning jobs with their details."""
        print(f"\n{'='*60}")
        print("FINE-TUNING JOBS")
        print(f"{'='*60}")
        
        try:
            jobs = self.client.fine_tuning.jobs.list(limit=limit).data
            
            if not jobs:
                print("No fine-tuning jobs found.")
                return
            
            print(f"{'CREATED':<20} {'STATUS':<12} {'JOB ID'}")
            print("-" * 60)
            
            for job in jobs:
                created = dt.datetime.fromtimestamp(job.created_at).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{created:<20} {job.status:<12} {job.id}")
                
        except Exception as e:
            print(f"Error listing jobs: {e}")
    
    def view_job(self, job_id: str) -> None:
        """View detailed information about a specific job."""
        print(f"\n{'='*60}")
        print(f"JOB DETAILS: {job_id}")
        print(f"{'='*60}")
        
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            print(job)
            
            # Basic job info
            created = dt.datetime.fromtimestamp(job.created_at).strftime("%Y-%m-%d %H:%M:%S")
            print(f"ID: {job.id}")
            print(f"Status: {job.status}")
            print(f"Created: {created}")
            print(f"Model: {job.model}")
            
            if job.fine_tuned_model:
                print(f"Fine-tuned Model: {job.fine_tuned_model}")
            
            # Usage metrics (only available when running or completed)
            usage_metrics = job.model_dump().get("usage_metrics")
            if usage_metrics:
                print(f"\nUSAGE METRICS:")
                
                # Check if this is a reinforcement learning job
                if usage_metrics.get('type') == 'reinforcement':
                    print(f"Type: Reinforcement Learning")
                    
                    # Training time
                    training_time = usage_metrics.get('training_time_seconds')
                    if training_time:
                        hours = training_time // 3600
                        minutes = (training_time % 3600) // 60
                        seconds = training_time % 60
                        print(f"Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
                    
                    # Model grader token usage
                    grader_usage = usage_metrics.get('model_grader_token_usage_per_model', {})
                    if grader_usage:
                        print(f"Grader Token Usage: {grader_usage}")
                    else:
                        print(f"Grader Token Usage: Not available yet")
                
                else:
                    # Standard supervised fine-tuning metrics
                    print(f"Type: Supervised Fine-tuning")
                    
                    # Format training tokens safely
                    training_tokens = usage_metrics.get('training_tokens')
                    if isinstance(training_tokens, (int, float)) and training_tokens is not None:
                        print(f"Training Tokens: {training_tokens:,}")
                    else:
                        print(f"Training Tokens: {training_tokens or 'N/A'}")
                    
                    # Format cost
                    cost = usage_metrics.get('cost_training_usd', 0)
                    print(f"Training Cost: ${cost:.4f}")
                    
                    # Format validation tokens safely
                    if 'validation_tokens' in usage_metrics:
                        validation_tokens = usage_metrics.get('validation_tokens')
                        if isinstance(validation_tokens, (int, float)) and validation_tokens is not None:
                            print(f"Validation Tokens: {validation_tokens:,}")
                        else:
                            print(f"Validation Tokens: {validation_tokens or 'N/A'}")
            else:
                print(f"\nJob is still '{job.status}'. No usage metrics available yet.")
            
            # Training files
            if hasattr(job, 'training_file') and job.training_file:
                print(f"\nTraining File: {job.training_file}")
            
            # Error info if failed
            if job.status == "failed" and hasattr(job, 'error') and job.error:
                print(f"\nError: {job.error}")
                
        except Exception as e:
            print(f"Error retrieving job: {e}")
    
    def list_events(self, job_id: str, limit: int = 20) -> None:
        """List events for a specific fine-tuning job."""
        print(f"\n{'='*60}")
        print(f"EVENTS FOR JOB: {job_id}")
        print(f"{'='*60}")
        
        try:
            events = self.client.fine_tuning.jobs.list_events(job_id, limit=limit).data
            
            if not events:
                print("No events found for this job.")
                return
            
            print(f"{'TIMESTAMP':<20} {'LEVEL':<8} {'MESSAGE'}")
            print("-" * 80)
            
            for event in events:
                timestamp = dt.datetime.fromtimestamp(event.created_at).strftime("%Y-%m-%d %H:%M:%S")
                level = event.level.upper() if hasattr(event, 'level') else "INFO"
                message = event.message
                print(f"{timestamp:<20} {level:<8} {message}")
                
            # Show full details for recent errors or warnings
            recent_issues = [e for e in events[:5] if hasattr(e, 'level') and e.level in ['error', 'warn']]
            if recent_issues:
                print(f"\n{'='*60}")
                print("RECENT ISSUES (Full Details):")
                print(f"{'='*60}")
                for event in recent_issues:
                    timestamp = dt.datetime.fromtimestamp(event.created_at).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{timestamp}] {event.level.upper()}:")
                    print(f"{event.message}")
                    
        except Exception as e:
            print(f"Error listing events: {e}")
    
    def view_job_detailed(self, job_id: str) -> None:
        """View detailed job information including events."""
        self.view_job(job_id)
        print(f"\n{'='*60}")
        print("RECENT EVENTS:")
        print(f"{'='*60}")
        self.list_events(job_id, limit=10)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific fine-tuning job."""
        try:
            # First check if job can be cancelled
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            
            if job.status in ("succeeded", "failed", "cancelled"):
                print(f"Job {job_id} is already {job.status} and cannot be cancelled.")
                return False
            
            # Cancel the job
            print(f"Cancelling job {job_id}...")
            cancelled_job = self.client.fine_tuning.jobs.cancel(job_id)
            print(f"Job {job_id} has been cancelled successfully.")
            return True
            
        except Exception as e:
            print(f"Error cancelling job: {e}")
            return False

    def monitor_job_with_budget(self, job_id: str, max_budget: float = 200.0) -> None:
        """Monitor a job and cancel if it exceeds the budget."""
        print(f"\n{'='*60}")
        print(f"MONITORING JOB: {job_id}")
        print(f"Budget Cap: ${max_budget:.2f}")
        print(f"{'='*60}")
        
        try:
            while True:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                current_time = dt.datetime.now().strftime("%H:%M:%S")
                
                print(f"[{current_time}] Status: {job.status}")
                
                # Check if job is finished
                if job.status in ("succeeded", "failed", "cancelled"):
                    print(f"Job finished with status: {job.status}")
                    break
                
                # Check budget if running
                if job.status == "running":
                    usage_metrics = getattr(job, "usage_metrics", None)
                    if usage_metrics:
                        # Handle reinforcement learning jobs
                        if usage_metrics.get('type') == 'reinforcement':
                            training_time = usage_metrics.get('training_time_seconds', 0)
                            hours = training_time / 3600
                            print(f"[{current_time}] Training time: {hours:.2f} hours")
                            # Note: Reinforcement learning jobs don't have direct cost metrics
                            # You might need to estimate based on time or other factors
                        else:
                            # Standard supervised fine-tuning
                            current_cost = usage_metrics.get('cost_training_usd', 0)
                            print(f"[{current_time}] Current cost: ${current_cost:.4f}")
                            
                            if current_cost >= max_budget:
                                print(f"💰 Budget cap of ${max_budget:.2f} reached!")
                                print("Cancelling job...")
                                self.cancel_job(job_id)
                                break
                
                # Wait before next check
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"Error monitoring job: {e}")


def main():
    """Main function with interactive menu."""
    manager = OpenAIJobManager()
    
    while True:
        print(f"\n{'='*40}")
        print("OpenAI Fine-tuning Job Manager")
        print(f"{'='*40}")
        print("1. List all jobs")
        print("2. View specific job")
        print("3. View job with events")
        print("4. List job events")
        print("5. Cancel job")
        print("6. Monitor job with budget")
        print("7. Exit")
        
        choice = input("\nSelect an option (1-7): ").strip()
        
        if choice == "1":
            limit = input("Enter limit (default 20): ").strip()
            limit = int(limit) if limit.isdigit() else 20
            manager.list_jobs(limit)
            
        elif choice == "2":
            job_id = input("Enter job ID: ").strip()
            if job_id:
                manager.view_job(job_id)
            else:
                print("Job ID cannot be empty.")
                
        elif choice == "3":
            job_id = input("Enter job ID: ").strip()
            if job_id:
                manager.view_job_detailed(job_id)
            else:
                print("Job ID cannot be empty.")
                
        elif choice == "4":
            job_id = input("Enter job ID: ").strip()
            if job_id:
                limit = input("Enter event limit (default 20): ").strip()
                limit = int(limit) if limit.isdigit() else 20
                manager.list_events(job_id, limit)
            else:
                print("Job ID cannot be empty.")
                
        elif choice == "5":
            job_id = input("Enter job ID to cancel: ").strip()
            if job_id:
                confirm = input(f"Are you sure you want to cancel {job_id}? (y/N): ").strip().lower()
                if confirm == 'y':
                    manager.cancel_job(job_id)
                else:
                    print("Cancellation aborted.")
            else:
                print("Job ID cannot be empty.")
                
        elif choice == "6":
            job_id = input("Enter job ID to monitor: ").strip()
            if job_id:
                budget = input("Enter budget cap in USD (default 200.0): ").strip()
                try:
                    budget = float(budget) if budget else 200.0
                    manager.monitor_job_with_budget(job_id, budget)
                except ValueError:
                    print("Invalid budget amount.")
            else:
                print("Job ID cannot be empty.")
                
        elif choice == "7":
            print("Goodbye!")
            break
            
        else:
            print("Invalid option. Please select 1-7.")


if __name__ == "__main__":
    # Quick usage examples (uncomment to use directly)
    # manager = OpenAIJobManager()
    # manager.list_jobs()
    # manager.view_job("your-job-id-here")
    # manager.cancel_job("your-job-id-here")
    
    # Run interactive menu
    main()