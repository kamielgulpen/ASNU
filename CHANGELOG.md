# Changelog

All notable changes to ASNU will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-27

### Added
- Initial release of ASNU (Aggregated Social Network Unfolder)
- Core network generation functionality with:
  - Community structure support
  - Preferential attachment mechanism
  - Reciprocity between nodes
  - Transitivity (friend-of-friend connections)
- Flexible input file support with configurable column names
- NetworkX-based graph representation
- Comprehensive examples for basic and custom usage
- Full test suite
- Documentation and README

### Features
- Generate population-based networks from aggregated data
- Support for both CSV and Excel input files
- Customizable network parameters (attachment, reciprocity, transitivity)
- Automatic network saving and metadata tracking
- Verbose mode for detailed progress reporting
