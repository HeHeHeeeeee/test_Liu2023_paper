"""
Metrics computation for VQA benchmarking.

Implements metrics from the paper:
1. Success rate and TTN (Total Trial Number)
2. Average iteration count
3. Gradient norm
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
import os


def compute_gradient_norm(gradients: np.ndarray) -> float:
    """
    Compute normalized gradient norm.
    
    From paper: G = (1/L) * Σ_l (∂C/∂θ_l)²
    
    Args:
        gradients: Gradient vector
        
    Returns:
        Normalized gradient norm
    """
    return np.mean(gradients ** 2)


def evaluate_success(final_energy: float, exact_energy: float, 
                    threshold: float = 1.6e-3) -> bool:
    """
    Check if optimization succeeded.
    
    Success criterion from paper: Within chemical accuracy (1.6e-3 Hartree).
    
    Args:
        final_energy: Optimized energy
        exact_energy: Exact ground state energy
        threshold: Chemical accuracy threshold
        
    Returns:
        True if successful
    """
    return abs(final_energy - exact_energy) < threshold


def compute_total_trial_number(results: List[Dict[str, Any]],
                               success_threshold: float = 1.6e-3,
                               target_successes: int = 100) -> int:
    """
    Compute TTN (Total Trial Number) to achieve target successes.
    
    From paper: number of trials needed to get 100 successful runs.
    
    Args:
        results: List of optimization results
        success_threshold: Threshold for success
        
    Returns:
        TTN value
    """
    successes = 0
    for idx, result in enumerate(results, start=1):
        if result.get("success", False) or result.get("error", float("inf")) < success_threshold:
            successes += 1
        if successes >= target_successes:
            return idx
    return len(results)


@dataclass
class MetricTracker:
    """Track and aggregate metrics across multiple runs."""
    
    task_name: str
    method_name: str
    exact_energy: float
    threshold: float = 1.6e-3
    
    # Raw data
    errors: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)
    gradient_norms: List[List[float]] = field(default_factory=list)
    cost_histories: List[List[float]] = field(default_factory=list)
    run_times: List[float] = field(default_factory=list)
    
    # Derived metrics
    _n_success: int = field(default=0, init=False)
    _ttn: int = field(default=0, init=False)
    
    def add_result(self, error: float, n_iterations: int, 
                   grad_norm_history: List[float],
                   cost_history: List[float],
                   run_time: float):
        """Add a single run result."""
        self.errors.append(error)
        self.iterations.append(n_iterations)
        self.gradient_norms.append(grad_norm_history)
        self.cost_histories.append(cost_history)
        self.run_times.append(run_time)
        
        if error < self.threshold:
            self._n_success += 1
        self._ttn += 1
    
    @property
    def n_trials(self) -> int:
        """Total number of trials."""
        return self._ttn
    
    @property
    def n_success(self) -> int:
        """Number of successful trials."""
        return self._n_success
    
    @property
    def success_rate(self) -> float:
        """Success rate."""
        if self._ttn == 0:
            return 0.0
        return self._n_success / self._ttn
    
    def get_avg_iterations(self) -> Tuple[float, float]:
        """
        Get average iterations for successful runs.
        
        Returns:
            Tuple of (mean, std)
        """
        successful_errors = [e for i, e in enumerate(self.errors) 
                           if e < self.threshold]
        successful_iters = [self.iterations[i] for i, e in enumerate(self.errors)
                          if e < self.threshold]
        
        if not successful_iters:
            return float('inf'), float('inf')
            
        return np.mean(successful_iters), np.std(successful_iters)
    
    def get_avg_gradient_norm(self) -> Tuple[float, float]:
        """
        Get average final gradient norm for successful runs.
        
        Returns:
            Tuple of (mean, std)
        """
        successful_grad_norms = [
            gn[-1] for i, gn in enumerate(self.gradient_norms)
            if self.errors[i] < self.threshold and len(gn) > 0
        ]
        
        if not successful_grad_norms:
            return 0.0, 0.0
            
        return np.mean(successful_grad_norms), np.std(successful_grad_norms)
    
    def get_avg_cost(self) -> Tuple[float, float]:
        """Get average final cost for successful runs."""
        successful_costs = [
            c[-1] for i, c in enumerate(self.cost_histories)
            if self.errors[i] < self.threshold and len(c) > 0
        ]
        
        if not successful_costs:
            return 0.0, 0.0
            
        return np.mean(successful_costs), np.std(successful_costs)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        avg_iter, std_iter = self.get_avg_iterations()
        avg_gn, std_gn = self.get_avg_gradient_norm()
        avg_cost, std_cost = self.get_avg_cost()
        
        return {
            'task': self.task_name,
            'method': self.method_name,
            'exact_energy': self.exact_energy,
            'n_trials': self.n_trials,
            'n_success': self.n_success,
            'success_rate': self.success_rate,
            'total_trial_number': self.n_trials,  # TTN
            'avg_iterations': avg_iter,
            'std_iterations': std_iter,
            'avg_gradient_norm': avg_gn,
            'std_gradient_norm': std_gn,
            'avg_final_cost': avg_cost,
            'std_final_cost': std_cost,
        }
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        summary = self.get_summary()
        
        # Include raw data for detailed analysis
        data = {
            'summary': summary,
            'errors': self.errors,
            'iterations': self.iterations,
            'gradient_norms': self.gradient_norms,
            'cost_histories': self.cost_histories,
            'run_times': self.run_times,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'MetricTracker':
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        summary = data['summary']
        tracker = cls(
            task_name=summary['task'],
            method_name=summary['method'],
            exact_energy=summary['exact_energy']
        )
        
        tracker.errors = data['errors']
        tracker.iterations = data['iterations']
        tracker.gradient_norms = data['gradient_norms']
        tracker.cost_histories = data['cost_histories']
        tracker.run_times = data['run_times']
        tracker._n_success = summary['n_success']
        tracker._ttn = summary['n_trials']
        
        return tracker


def compare_methods(all_metrics: Dict[str, MetricTracker]) -> Dict[str, Any]:
    """
    Compare results across different initialization methods.
    
    Args:
        all_metrics: Dict of method_name -> MetricTracker
        
    Returns:
        Comparison results
    """
    comparison = {}
    
    for method, metrics in all_metrics.items():
        summary = metrics.get_summary()
        comparison[method] = {
            'TTN': summary['total_trial_number'],
            'avg_iterations': summary['avg_iterations'],
            'std_iterations': summary['std_iterations'],
            'success_rate': summary['success_rate'],
            'avg_gradient_norm': summary['avg_gradient_norm'],
        }
    
    return comparison


def create_comparison_table(all_metrics: Dict[str, MetricTracker]) -> str:
    """
    Create a formatted comparison table (similar to paper Table 1).
    
    Args:
        all_metrics: Dict of method_name -> MetricTracker
        
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Method':<15} {'TTN':<10} {'Avg Iter':<15} {'Std Iter':<12} {'Grad Norm':<12}")
    lines.append("=" * 80)
    
    for method, metrics in all_metrics.items():
        summary = metrics.get_summary()
        lines.append(
            f"{method:<15} "
            f"{summary['total_trial_number']:<10} "
            f"{summary['avg_iterations']:<15.2f} "
            f"{summary['std_iterations']:<12.2f} "
            f"{summary['avg_gradient_norm']:<12.6f}"
        )
    
    lines.append("=" * 80)
    return '\n'.join(lines)
