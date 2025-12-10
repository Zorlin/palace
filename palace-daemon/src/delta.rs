//! Delta parsing for the ++/-- context protocol
//!
//! The 1b classifier outputs simple add/remove instructions:
//! ```text
//! ++1,2,3,4--5,6,7,8
//! ```
//!
//! This module parses those instructions into structured deltas.

use crate::{BlockId, DeltaError};

/// A parsed delta instruction
#[derive(Debug, Clone, Default)]
pub struct Delta {
    /// Block IDs to add to active set
    pub add: Vec<BlockId>,
    /// Block IDs to remove from active set
    pub remove: Vec<BlockId>,
}

/// Parse a delta string into structured add/remove lists
///
/// Supported formats:
/// - `++1,2,3--4,5,6` - Add 1,2,3 and remove 4,5,6
/// - `++1,2,3` - Add only
/// - `--1,2,3` - Remove only
/// - `++1--2++3--4` - Interleaved (order preserved)
pub fn parse_delta(input: &str) -> Result<Delta, DeltaError> {
    let input = input.trim();

    if input.is_empty() {
        return Ok(Delta::default());
    }

    let mut delta = Delta::default();
    let mut pos = 0;
    let chars: Vec<char> = input.chars().collect();

    while pos < chars.len() {
        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        if pos >= chars.len() {
            break;
        }

        // Check for ++ or --
        if pos + 1 < chars.len() {
            let op = format!("{}{}", chars[pos], chars[pos + 1]);

            match op.as_str() {
                "++" => {
                    pos += 2;
                    let ids = parse_id_list(&chars, &mut pos)?;
                    delta.add.extend(ids);
                }
                "--" => {
                    pos += 2;
                    let ids = parse_id_list(&chars, &mut pos)?;
                    delta.remove.extend(ids);
                }
                _ => {
                    return Err(DeltaError::InvalidFormat(format!(
                        "Expected ++ or -- at position {}, got '{}'",
                        pos, op
                    )));
                }
            }
        } else {
            return Err(DeltaError::InvalidFormat(format!(
                "Unexpected end of input at position {}",
                pos
            )));
        }
    }

    Ok(delta)
}

/// Parse a comma-separated list of IDs
fn parse_id_list(chars: &[char], pos: &mut usize) -> Result<Vec<BlockId>, DeltaError> {
    let mut ids = Vec::new();
    let mut current_num = String::new();

    while *pos < chars.len() {
        let c = chars[*pos];

        if c.is_ascii_digit() {
            current_num.push(c);
            *pos += 1;
        } else if c == ',' {
            if !current_num.is_empty() {
                let id: BlockId = current_num
                    .parse()
                    .map_err(|e| DeltaError::ParseError(format!("Invalid ID '{}': {}", current_num, e)))?;
                ids.push(id);
                current_num.clear();
            }
            *pos += 1;
        } else if c == '+' || c == '-' {
            // Hit next operator
            break;
        } else if c.is_whitespace() {
            *pos += 1;
        } else {
            return Err(DeltaError::InvalidFormat(format!(
                "Unexpected character '{}' at position {}",
                c, pos
            )));
        }
    }

    // Don't forget the last number
    if !current_num.is_empty() {
        let id: BlockId = current_num
            .parse()
            .map_err(|e| DeltaError::ParseError(format!("Invalid ID '{}': {}", current_num, e)))?;
        ids.push(id);
    }

    Ok(ids)
}

/// Format a delta back to string representation
pub fn format_delta(delta: &Delta) -> String {
    let mut result = String::new();

    if !delta.add.is_empty() {
        result.push_str("++");
        result.push_str(
            &delta
                .add
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }

    if !delta.remove.is_empty() {
        result.push_str("--");
        result.push_str(
            &delta
                .remove
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_add_only() {
        let delta = parse_delta("++1,2,3").unwrap();
        assert_eq!(delta.add, vec![1, 2, 3]);
        assert!(delta.remove.is_empty());
    }

    #[test]
    fn test_parse_remove_only() {
        let delta = parse_delta("--4,5,6").unwrap();
        assert!(delta.add.is_empty());
        assert_eq!(delta.remove, vec![4, 5, 6]);
    }

    #[test]
    fn test_parse_both() {
        let delta = parse_delta("++1,2,3--4,5,6").unwrap();
        assert_eq!(delta.add, vec![1, 2, 3]);
        assert_eq!(delta.remove, vec![4, 5, 6]);
    }

    #[test]
    fn test_parse_interleaved() {
        let delta = parse_delta("++1--2++3--4").unwrap();
        assert_eq!(delta.add, vec![1, 3]);
        assert_eq!(delta.remove, vec![2, 4]);
    }

    #[test]
    fn test_parse_with_spaces() {
        let delta = parse_delta("  ++1, 2, 3  --4, 5  ").unwrap();
        assert_eq!(delta.add, vec![1, 2, 3]);
        assert_eq!(delta.remove, vec![4, 5]);
    }

    #[test]
    fn test_parse_single_ids() {
        let delta = parse_delta("++1--2").unwrap();
        assert_eq!(delta.add, vec![1]);
        assert_eq!(delta.remove, vec![2]);
    }

    #[test]
    fn test_parse_empty() {
        let delta = parse_delta("").unwrap();
        assert!(delta.add.is_empty());
        assert!(delta.remove.is_empty());
    }

    #[test]
    fn test_format_delta() {
        let delta = Delta {
            add: vec![1, 2, 3],
            remove: vec![4, 5, 6],
        };
        assert_eq!(format_delta(&delta), "++1,2,3--4,5,6");
    }

    #[test]
    fn test_format_add_only() {
        let delta = Delta {
            add: vec![1, 2],
            remove: vec![],
        };
        assert_eq!(format_delta(&delta), "++1,2");
    }

    #[test]
    fn test_roundtrip() {
        let original = "++1,2,3--4,5,6";
        let delta = parse_delta(original).unwrap();
        let formatted = format_delta(&delta);
        assert_eq!(formatted, original);
    }

    #[test]
    fn test_invalid_format() {
        assert!(parse_delta("1,2,3").is_err()); // No operator
        assert!(parse_delta("+1").is_err()); // Single +
    }
}
