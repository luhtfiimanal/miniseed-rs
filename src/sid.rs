//! FDSN Source Identifier (SID) for miniSEED v3.
//!
//! The FDSN Source Identifier uses the format `FDSN:NET_STA_LOC_BAND_SOURCE_SS`
//! where the channel code (e.g. "BHZ") is split into band ("B"), source ("H"),
//! and subsource ("Z") components.

use std::fmt;

/// FDSN Source Identifier.
///
/// Format: `FDSN:NET_STA_LOC_BAND_SOURCE_SUBSOURCE`
///
/// # Examples
///
/// ```
/// use miniseed_rs::SourceId;
///
/// let sid = SourceId::from_nslc("IU", "ANMO", "00", "BHZ");
/// assert_eq!(sid.as_str(), "FDSN:IU_ANMO_00_B_H_Z");
/// assert_eq!(sid.network(), "IU");
/// assert_eq!(sid.station(), "ANMO");
/// assert_eq!(sid.location(), "00");
/// assert_eq!(sid.channel(), "BHZ");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceId {
    raw: String,
}

impl SourceId {
    /// Parse a source identifier string.
    ///
    /// Accepts either the full `FDSN:` prefix form or just the underscore-separated
    /// components. Returns the canonical `FDSN:` prefixed form.
    pub fn parse(s: &str) -> Self {
        let raw = if s.starts_with("FDSN:") {
            s.to_string()
        } else {
            format!("FDSN:{s}")
        };
        Self { raw }
    }

    /// Create a source identifier from NSLC codes.
    ///
    /// The 3-character channel code is split into band (1 char),
    /// source (1 char), and subsource (1 char).
    pub fn from_nslc(network: &str, station: &str, location: &str, channel: &str) -> Self {
        let (band, source, subsource) = split_channel(channel);
        Self {
            raw: format!("FDSN:{network}_{station}_{location}_{band}_{source}_{subsource}"),
        }
    }

    /// Return the raw source identifier string.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// Extract the network code.
    pub fn network(&self) -> &str {
        self.component(0)
    }

    /// Extract the station code.
    pub fn station(&self) -> &str {
        self.component(1)
    }

    /// Extract the location code.
    pub fn location(&self) -> &str {
        self.component(2)
    }

    /// Extract the reconstructed channel code (band + source + subsource).
    pub fn channel(&self) -> String {
        let band = self.component(3);
        let source = self.component(4);
        let subsource = self.component(5);
        format!("{band}{source}{subsource}")
    }

    /// Extract the NSLC tuple: (network, station, location, channel).
    pub fn to_nslc(&self) -> (String, String, String, String) {
        (
            self.network().to_string(),
            self.station().to_string(),
            self.location().to_string(),
            self.channel(),
        )
    }

    /// Get the Nth underscore-separated component after the "FDSN:" prefix.
    fn component(&self, index: usize) -> &str {
        let body = self.raw.strip_prefix("FDSN:").unwrap_or(&self.raw);
        body.split('_').nth(index).unwrap_or("")
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

/// Split a 3-character channel code into (band, source, subsource).
///
/// For shorter codes, missing components default to empty strings.
fn split_channel(channel: &str) -> (&str, &str, &str) {
    let chars: Vec<char> = channel.chars().collect();
    match chars.len() {
        0 => ("", "", ""),
        1 => (&channel[0..1], "", ""),
        2 => (&channel[0..1], &channel[1..2], ""),
        _ => (&channel[0..1], &channel[1..2], &channel[2..3]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_nslc() {
        let sid = SourceId::from_nslc("IU", "ANMO", "00", "BHZ");
        assert_eq!(sid.as_str(), "FDSN:IU_ANMO_00_B_H_Z");
    }

    #[test]
    fn test_parse_with_prefix() {
        let sid = SourceId::parse("FDSN:IU_ANMO_00_B_H_Z");
        assert_eq!(sid.as_str(), "FDSN:IU_ANMO_00_B_H_Z");
    }

    #[test]
    fn test_parse_without_prefix() {
        let sid = SourceId::parse("IU_ANMO_00_B_H_Z");
        assert_eq!(sid.as_str(), "FDSN:IU_ANMO_00_B_H_Z");
    }

    #[test]
    fn test_component_extraction() {
        let sid = SourceId::from_nslc("IU", "ANMO", "00", "BHZ");
        assert_eq!(sid.network(), "IU");
        assert_eq!(sid.station(), "ANMO");
        assert_eq!(sid.location(), "00");
        assert_eq!(sid.channel(), "BHZ");
    }

    #[test]
    fn test_to_nslc() {
        let sid = SourceId::from_nslc("GE", "DAV", "10", "HHE");
        let (net, sta, loc, cha) = sid.to_nslc();
        assert_eq!(net, "GE");
        assert_eq!(sta, "DAV");
        assert_eq!(loc, "10");
        assert_eq!(cha, "HHE");
    }

    #[test]
    fn test_empty_location() {
        let sid = SourceId::from_nslc("JP", "TSK", "", "LHN");
        assert_eq!(sid.as_str(), "FDSN:JP_TSK__L_H_N");
        assert_eq!(sid.location(), "");
        assert_eq!(sid.channel(), "LHN");
    }

    #[test]
    fn test_display() {
        let sid = SourceId::from_nslc("IU", "ANMO", "00", "BHZ");
        assert_eq!(format!("{sid}"), "FDSN:IU_ANMO_00_B_H_Z");
    }

    #[test]
    fn test_roundtrip_nslc() {
        let original = SourceId::from_nslc("XX", "TEST", "00", "BHZ");
        let (net, sta, loc, cha) = original.to_nslc();
        let reconstructed = SourceId::from_nslc(&net, &sta, &loc, &cha);
        assert_eq!(original, reconstructed);
    }
}
