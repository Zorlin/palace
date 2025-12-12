//! Interactive action menu for Palace CLI
//!
//! Presents Claude's suggested actions to the user and handles selection.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

/// A suggested action from Claude
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub label: String,
    pub description: Option<String>,
}

/// Present an interactive action menu to the user
pub fn present_action_menu(actions: &[Action], simple_menu: bool) -> Result<Option<Vec<Action>>> {
    if actions.is_empty() {
        println!("\n📋 No actions suggested.");
        return Ok(None);
    }

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════");
    println!("                     📋 SUGGESTED ACTIONS");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Display actions
    for (i, action) in actions.iter().enumerate() {
        let num = i + 1;
        if simple_menu {
            println!("  {:>2}. {}", num, action.label);
        } else {
            println!("  {:>2}. {}", num, action.label);
            if let Some(desc) = &action.description {
                // Wrap description at 70 chars
                let wrapped = textwrap(desc, 70, "      ");
                println!("{}", wrapped);
            }
            println!();
        }
    }

    println!("───────────────────────────────────────────────────────────────");
    println!();
    println!("  Selection options:");
    println!("    • Enter numbers: 1,2,3 or 1 2 3");
    println!("    • Enter range: 1-5");
    println!("    • Enter 'all' to select all");
    println!("    • Enter 'q' or press Enter to quit");
    println!();
    print!("  Select actions: ");
    io::stdout().flush()?;

    // Read user input
    let mut input = String::new();
    io::stdin().lock().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() || input.eq_ignore_ascii_case("q") {
        println!("\n👋 No actions selected.");
        return Ok(None);
    }

    // Parse selection
    let selected = parse_selection(input, actions.len());

    if selected.is_empty() {
        println!("\n⚠️  Invalid selection.");
        return Ok(None);
    }

    let selected_actions: Vec<Action> = selected
        .into_iter()
        .map(|i| actions[i].clone())
        .collect();

    println!();
    println!("✅ Selected {} action(s):", selected_actions.len());
    for action in &selected_actions {
        println!("   • {}", action.label);
    }
    println!();

    Ok(Some(selected_actions))
}

/// Parse user selection input
fn parse_selection(input: &str, max: usize) -> Vec<usize> {
    let input = input.to_lowercase();

    // Handle "all"
    if input == "all" {
        return (0..max).collect();
    }

    let mut indices = Vec::new();

    // Split by comma or space
    for part in input.split(|c| c == ',' || c == ' ') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Check for range (e.g., "1-5")
        if part.contains('-') {
            let parts: Vec<&str> = part.split('-').collect();
            if parts.len() == 2 {
                if let (Ok(start), Ok(end)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                {
                    for i in start..=end {
                        if i >= 1 && i <= max {
                            indices.push(i - 1);
                        }
                    }
                }
            }
        } else if let Ok(num) = part.parse::<usize>() {
            if num >= 1 && num <= max {
                indices.push(num - 1);
            }
        }
    }

    // Remove duplicates while preserving order
    let mut seen = std::collections::HashSet::new();
    indices.retain(|x| seen.insert(*x));

    indices
}

/// Simple text wrapping
fn textwrap(text: &str, width: usize, prefix: &str) -> String {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.len() + 1 + word.len() <= width {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(format!("{}{}", prefix, current_line));
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(format!("{}{}", prefix, current_line));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_selection_single() {
        assert_eq!(parse_selection("1", 5), vec![0]);
        assert_eq!(parse_selection("3", 5), vec![2]);
    }

    #[test]
    fn test_parse_selection_multiple() {
        assert_eq!(parse_selection("1,2,3", 5), vec![0, 1, 2]);
        assert_eq!(parse_selection("1 2 3", 5), vec![0, 1, 2]);
        assert_eq!(parse_selection("1, 2, 3", 5), vec![0, 1, 2]);
    }

    #[test]
    fn test_parse_selection_range() {
        assert_eq!(parse_selection("1-3", 5), vec![0, 1, 2]);
        assert_eq!(parse_selection("2-4", 5), vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_selection_all() {
        assert_eq!(parse_selection("all", 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(parse_selection("ALL", 3), vec![0, 1, 2]);
    }

    #[test]
    fn test_parse_selection_out_of_bounds() {
        let empty: Vec<usize> = vec![];
        assert_eq!(parse_selection("10", 5), empty);
        assert_eq!(parse_selection("0", 5), empty);
    }

    #[test]
    fn test_parse_selection_deduplication() {
        assert_eq!(parse_selection("1,1,2,2", 5), vec![0, 1]);
    }

    #[test]
    fn test_textwrap() {
        let text = "This is a long line that should be wrapped at a certain width.";
        let wrapped = textwrap(text, 20, "  ");
        assert!(wrapped.contains("\n"));
        assert!(wrapped.starts_with("  "));
    }
}
